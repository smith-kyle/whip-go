package main

import (
	"errors"
	"fmt"
	"strings"

	"github.com/pion/mediadevices/pkg/codec"
	"github.com/pion/mediadevices/pkg/io/audio"
	"github.com/pion/mediadevices/pkg/io/video"
	"github.com/pion/mediadevices/pkg/prop"
	"github.com/pion/webrtc/v3"
)

// CodecSelector is a container of video and audio encoder builders, which later will be used
// for codec matching.
type CodecSelector struct {
	videoEncoders []codec.VideoEncoderBuilder
	audioEncoders []codec.AudioEncoderBuilder
}

// CodecSelectorOption is a type for specifying CodecSelector options
type CodecSelectorOption func(*CodecSelector)

// WithVideoEncoders replace current video codecs with listed encoders
func WithVideoEncoders(encoders ...codec.VideoEncoderBuilder) CodecSelectorOption {
	return func(t *CodecSelector) {
		t.videoEncoders = encoders
	}
}

// WithVideoEncoders replace current audio codecs with listed encoders
func WithAudioEncoders(encoders ...codec.AudioEncoderBuilder) CodecSelectorOption {
	return func(t *CodecSelector) {
		t.audioEncoders = encoders
	}
}

// NewCodecSelector constructs CodecSelector with given variadic options
func NewCodecSelector(opts ...CodecSelectorOption) *CodecSelector {
	var track CodecSelector

	for _, opt := range opts {
		opt(&track)
	}

	return &track
}

// Populate lets the webrtc engine be aware of supported codecs that are contained in CodecSelector
func (selector *CodecSelector) Populate(setting *webrtc.MediaEngine) {
	for _, encoder := range selector.videoEncoders {
		setting.RegisterCodec(encoder.RTPCodec().RTPCodecParameters, webrtc.RTPCodecTypeVideo)
	}

	for _, encoder := range selector.audioEncoders {
		setting.RegisterCodec(encoder.RTPCodec().RTPCodecParameters, webrtc.RTPCodecTypeAudio)
	}
}

// selectVideoCodecByNames selects a single codec that can be built and matched. codecNames can be formatted as "video/<codecName>" or "<codecName>"
func (selector *CodecSelector) selectVideoCodecByNames(reader video.Reader, inputProp prop.Media, codecNames ...string) (codec.ReadCloser, *codec.RTPCodec, error) {
	var selectedEncoder codec.VideoEncoderBuilder
	var encodedReader codec.ReadCloser
	var errReasons []string
	var err error

outer:
	for _, wantCodec := range codecNames {
		wantCodecLower := strings.ToLower(wantCodec)
		for _, encoder := range selector.videoEncoders {
			// MimeType is formated as "video/<codecName>"
			if strings.HasSuffix(strings.ToLower(encoder.RTPCodec().MimeType), wantCodecLower) {
				encodedReader, err = encoder.BuildVideoEncoder(reader, inputProp)
				if err == nil {
					selectedEncoder = encoder
					break outer
				}
			}

			errReasons = append(errReasons, fmt.Sprintf("%s: %s", encoder.RTPCodec().MimeType, err))
		}
	}

	if selectedEncoder == nil {
		return nil, nil, errors.New(strings.Join(errReasons, "\n\n"))
	}

	return encodedReader, selectedEncoder.RTPCodec(), nil
}

func (selector *CodecSelector) selectVideoCodec(reader video.Reader, inputProp prop.Media, codecs ...webrtc.RTPCodecParameters) (codec.ReadCloser, *codec.RTPCodec, error) {
	var codecNames []string

	for _, codec := range codecs {
		codecNames = append(codecNames, codec.MimeType)
	}

	return selector.selectVideoCodecByNames(reader, inputProp, codecNames...)
}

// selectAudioCodecByNames selects a single codec that can be built and matched. codecNames can be formatted as "audio/<codecName>" or "<codecName>"
func (selector *CodecSelector) selectAudioCodecByNames(reader audio.Reader, inputProp prop.Media, codecNames ...string) (codec.ReadCloser, *codec.RTPCodec, error) {
	var selectedEncoder codec.AudioEncoderBuilder
	var encodedReader codec.ReadCloser
	var errReasons []string
	var err error

outer:
	for _, wantCodec := range codecNames {
		wantCodecLower := strings.ToLower(wantCodec)
		for _, encoder := range selector.audioEncoders {
			// MimeType is formated as "audio/<codecName>"
			if strings.HasSuffix(strings.ToLower(encoder.RTPCodec().MimeType), wantCodecLower) {
				encodedReader, err = encoder.BuildAudioEncoder(reader, inputProp)
				if err == nil {
					selectedEncoder = encoder
					break outer
				}
			}

			errReasons = append(errReasons, fmt.Sprintf("%s: %s", encoder.RTPCodec().MimeType, err))
		}
	}

	if selectedEncoder == nil {
		return nil, nil, errors.New(strings.Join(errReasons, "\n\n"))
	}

	return encodedReader, selectedEncoder.RTPCodec(), nil
}

func (selector *CodecSelector) selectAudioCodec(reader audio.Reader, inputProp prop.Media, codecs ...webrtc.RTPCodecParameters) (codec.ReadCloser, *codec.RTPCodec, error) {
	var codecNames []string

	for _, codec := range codecs {
		codecNames = append(codecNames, codec.MimeType)
	}

	return selector.selectAudioCodecByNames(reader, inputProp, codecNames...)
}

// H264PassthroughCodec creates RTP codec parameters for H.264 passthrough
func H264PassthroughCodec() webrtc.RTPCodecParameters {
	return webrtc.RTPCodecParameters{
		RTPCodecCapability: webrtc.RTPCodecCapability{
			MimeType:    "video/H264",
			ClockRate:   90000,
			Channels:    0,
			SDPFmtpLine: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=640028",
		},
		PayloadType: 96,
	}
}

// WithH264PassthroughEncoder adds H.264 passthrough support to codec selector
func WithH264PassthroughEncoder() CodecSelectorOption {
	return func(selector *CodecSelector) {
		// Add H.264 passthrough "encoder" (just codec parameters)
		selector.videoEncoders = append(selector.videoEncoders, &H264PassthroughEncoder{})
	}
}

// H264PassthroughEncoder implements codec.VideoEncoderBuilder for H.264 passthrough
type H264PassthroughEncoder struct{}

func (e *H264PassthroughEncoder) RTPCodec() *codec.RTPCodec {
	return &codec.RTPCodec{
		RTPCodecParameters: H264PassthroughCodec(),
		Payloader:          &H264Payloader{},
		Latency:            0, // No latency for passthrough
	}
}

func (e *H264PassthroughEncoder) BuildVideoEncoder(r video.Reader, p prop.Media) (codec.ReadCloser, error) {
	// This should not be called for H.264 passthrough mode
	return nil, errors.New("BuildVideoEncoder should not be called for H.264 passthrough mode")
}
