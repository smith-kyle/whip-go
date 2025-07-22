package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"io"
	"log"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/pion/interceptor"
	"github.com/pion/mediadevices"
	"github.com/pion/mediadevices/pkg/codec"
	"github.com/pion/mediadevices/pkg/io/audio"
	"github.com/pion/mediadevices/pkg/io/video"
	"github.com/pion/mediadevices/pkg/prop"
	"github.com/pion/mediadevices/pkg/wave"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v3"
)

func GetInputMediaStream(audio string, video string, codecSelector *CodecSelector) (mediadevices.MediaStream, error) {
	tracks := make([]mediadevices.Track, 0)

	if len(audio) > 0 {
		track, err := GetAudioTrack(audio, codecSelector)
		if err != nil {
			return nil, err
		}
		tracks = append(tracks, track)
	}

	if len(video) > 0 {
		track, err := GetVideoTrack(video, codecSelector)
		if err != nil {
			return nil, err
		}
		tracks = append(tracks, track)
	}

	stream, err := mediadevices.NewMediaStream(tracks...)
	if err != nil {
		return nil, err
	}

	return stream, nil
}

func GetAudioTrack(name string, codecSelector *CodecSelector) (mediadevices.Track, error) {
	pipe, _ := os.Open(name)
	data := make([]byte, 480*2)
	chunkInfo := wave.ChunkInfo{
		Len:          480,
		Channels:     1,
		SamplingRate: 48000,
	}

	reader := audio.ReaderFunc(func() (chunk wave.Audio, release func(), err error) {
		_, err = io.ReadFull(pipe, data)
		buffer := wave.NewInt16Interleaved(chunkInfo)
		binary.Read(bytes.NewReader(data), binary.LittleEndian, buffer.Data)
		chunk = buffer
		return chunk, func() {}, err
	})
	track := newAudioTrackFromReader(reader, codecSelector)
	return track, nil
}

func GetVideoTrack(name string, codecSelector *CodecSelector) (mediadevices.Track, error) {
	pipe, _ := os.Open(name)
	area := 1280 * 720
	data := make([]byte, 1280*720*1.5)

	reader := video.ReaderFunc(func() (img image.Image, release func(), err error) {
		_, err = io.ReadFull(pipe, data)
		yuv := image.NewYCbCr(image.Rect(0, 0, 1280, 720), image.YCbCrSubsampleRatio420)

		copy(yuv.Y, data[0:area])
		copy(yuv.Cb, data[area:area+area/4])
		copy(yuv.Cr, data[area+area/4:area+area/4+area/4])
		img = yuv
		return img, func() {}, err
	})
	track := newVideoTrackFromReader(reader, codecSelector)
	return track, nil
}

func GetH264StreamFromStdin(codecSelector *CodecSelector, framerate int) (mediadevices.MediaStream, error) {
	log.Printf("Initializing H.264 stream from stdin at %d fps...", framerate)
	track, err := GetH264VideoTrackFromStdin(codecSelector, framerate)
	if err != nil {
		return nil, err
	}

	stream, err := mediadevices.NewMediaStream(track)
	if err != nil {
		return nil, err
	}

	log.Printf("H.264 stream initialized successfully")
	return stream, nil
}

func GetH264VideoTrackFromStdin(codecSelector *CodecSelector, framerate int) (mediadevices.Track, error) {
	track := newH264VideoTrackFromStdin(codecSelector, framerate)
	return track, nil
}

func newH264VideoTrackFromStdin(codecSelector *CodecSelector, framerate int) mediadevices.Track {
	base := newBaseTrack(mediadevices.VideoInput, codecSelector)

	return &H264VideoTrack{
		baseTrack: base,
		framerate: framerate,
	}
}

type H264VideoTrack struct {
	*baseTrack
	framerate int
}

func (track *H264VideoTrack) Bind(ctx webrtc.TrackLocalContext) (webrtc.RTPCodecParameters, error) {
	return track.bind(ctx, track)
}

func (track *H264VideoTrack) Unbind(ctx webrtc.TrackLocalContext) error {
	return track.unbind(ctx)
}

func (track *H264VideoTrack) NewEncodedReader(codecName string) (mediadevices.EncodedReadCloser, error) {
	if !strings.Contains(strings.ToLower(codecName), "h264") {
		return nil, errors.New("H264VideoTrack only supports H.264 codec")
	}

	return &h264StdinReader{}, nil
}

func (track *H264VideoTrack) NewEncodedIOReader(codecName string) (io.ReadCloser, error) {
	return nil, errors.New("H264VideoTrack does not support NewEncodedIOReader")
}

func (track *H264VideoTrack) NewRTPReader(codecName string, ssrc uint32, mtu int) (mediadevices.RTPReadCloser, error) {
	if !strings.Contains(strings.ToLower(codecName), "h264") {
		return nil, errors.New("H264VideoTrack only supports H.264 codec")
	}

	h264Codec := codec.NewRTPH264Codec(90000)
	// Set profile-level-id to 42c01f (constrained baseline, level 3.1)
	if h264Codec.RTPCodecParameters.SDPFmtpLine == "" {
		h264Codec.RTPCodecParameters.SDPFmtpLine = "profile-level-id=42c01f"
	} else {
		h264Codec.RTPCodecParameters.SDPFmtpLine += ";profile-level-id=42c01f"
	}

	packetizer := rtp.NewPacketizer(uint16(mtu), uint8(h264Codec.PayloadType), ssrc, h264Codec.Payloader, rtp.NewRandomSequencer(), h264Codec.ClockRate)

	return &h264RTPReader{
		packetizer: packetizer,
		stdinReader: &h264StdinReader{
			framerate: track.framerate,
			sampler:   newFixedVideoSampler(h264Codec.ClockRate, track.framerate),
		},
	}, nil
}

type h264StdinReader struct {
	closed        bool
	buffer        []byte
	frameCount    int
	lastFrameTime time.Time
	framerate     int
	sampler       samplerFunc
}

func (r *h264StdinReader) Read() (mediadevices.EncodedBuffer, func(), error) {
	if r.closed {
		return mediadevices.EncodedBuffer{}, func() {}, io.EOF
	}

	// Read a larger chunk to get complete NAL units
	chunk := make([]byte, 65536)
	n, err := os.Stdin.Read(chunk)
	if err != nil {
		if err == io.EOF {
			log.Printf("H.264 stream ended (EOF)")
			r.closed = true
		} else {
			log.Printf("Error reading H.264 stream: %v", err)
		}
		return mediadevices.EncodedBuffer{}, func() {}, err
	}

	if n == 0 {
		log.Printf("H.264 stream ended (0 bytes)")
		r.closed = true
		return mediadevices.EncodedBuffer{}, func() {}, io.EOF
	}

	// Append to buffer
	r.buffer = append(r.buffer, chunk[:n]...)

	// VERIFICATION: Count NAL units in the newly read data
	nalCount := r.countNALUnitsInRange(len(r.buffer)-n, len(r.buffer))
	if nalCount > 0 {
		log.Printf("STDIN READ: %d bytes, found %d NAL units in this read", n, nalCount)
	}

	for {
		// Find NAL unit boundaries
		nalStart := r.findNextNALUnit(0)
		if nalStart == -1 {
			// No complete NAL unit yet, need more data
			return r.Read() // Only recurse when buffer is truly empty
		}

		nalEnd := r.findNextNALUnit(nalStart + 4)
		if nalEnd == -1 {
			nalEnd = len(r.buffer)
		}

		nalData := make([]byte, nalEnd-nalStart)
		copy(nalData, r.buffer[nalStart:nalEnd])

		// Remove this NAL unit from buffer
		r.buffer = r.buffer[nalEnd:]

		// Check if this is an AUD (type 9)
		if len(nalData) >= 5 {
			nalType := nalData[4] & 0x1F
			if nalType == 9 {
				continue
			}
		}

		totalNalCount := r.countNALUnitsInRange(0, len(r.buffer))
		log.Printf("Total NAL units in buffer after processing: %d", totalNalCount)

		r.frameCount++
		now := time.Now()
		if r.frameCount%30 == 1 { // Log every 30 frames (~1 second at 30fps)
			fps := 0.0
			if r.frameCount > 1 && !r.lastFrameTime.IsZero() {
				elapsed := now.Sub(r.lastFrameTime)
				fps = 29.0 / elapsed.Seconds() // 29 frames processed in elapsed time
			}
			log.Printf("H.264 frame %d: %d bytes (buffer: %d bytes) - FPS: %.1f", r.frameCount, len(nalData), len(r.buffer), fps)
			r.lastFrameTime = now
		}

		// Use dynamic timestamp calculation like the existing video track
		samples := r.sampler()

		encoded := mediadevices.EncodedBuffer{
			Data:    nalData,
			Samples: samples,
		}
		return encoded, func() {}, nil
	}
}

func (r *h264StdinReader) findNextNALUnit(start int) int {
	if len(r.buffer) < start+4 {
		return -1
	}

	for i := start; i < len(r.buffer)-3; i++ {
		// Prioritize 4-byte start code
		if r.buffer[i] == 0x00 && r.buffer[i+1] == 0x00 && r.buffer[i+2] == 0x00 && r.buffer[i+3] == 0x01 {
			return i
		}
	}

	// Only check 3-byte if no 4-byte found
	for i := start; i < len(r.buffer)-2; i++ {
		if r.buffer[i] == 0x00 && r.buffer[i+1] == 0x00 && r.buffer[i+2] == 0x01 {
			return i
		}
	}
	return -1
}

func (r *h264StdinReader) countNALUnitsInRange(start, end int) int {
	count := 0
	pos := start

	for pos < end-3 {
		// Check for 4-byte start code
		if pos <= end-4 && r.buffer[pos] == 0x00 && r.buffer[pos+1] == 0x00 &&
			r.buffer[pos+2] == 0x00 && r.buffer[pos+3] == 0x01 {
			count++
			pos += 4
			continue
		}
		// Check for 3-byte start code
		if pos <= end-3 && r.buffer[pos] == 0x00 && r.buffer[pos+1] == 0x00 &&
			r.buffer[pos+2] == 0x01 {
			count++
			pos += 3
			continue
		}
		pos++
	}

	return count
}

func (r *h264StdinReader) Close() error {
	r.closed = true
	return nil
}

func (r *h264StdinReader) Controller() codec.EncoderController {
	return nil
}

type h264RTPReader struct {
	packetizer  rtp.Packetizer
	stdinReader *h264StdinReader
}

func (r *h264RTPReader) Read() ([]*rtp.Packet, func(), error) {
	encoded, release, err := r.stdinReader.Read()
	if err != nil {
		return nil, func() {}, err
	}
	defer release()

	pkts := r.packetizer.Packetize(encoded.Data, encoded.Samples)
	return pkts, release, nil
}

func (r *h264RTPReader) Close() error {
	return r.stdinReader.Close()
}

func (r *h264RTPReader) Controller() codec.EncoderController {
	return nil
}

type baseTrack struct {
	mediadevices.Source
	err                   error
	onErrorHandler        func(error)
	mu                    sync.Mutex
	endOnce               sync.Once
	kind                  mediadevices.MediaDeviceType
	selector              *CodecSelector
	activePeerConnections map[string]chan<- chan<- struct{}
}

func newBaseTrack(kind mediadevices.MediaDeviceType, selector *CodecSelector) *baseTrack {
	return &baseTrack{
		Source:                NewSource(),
		kind:                  kind,
		selector:              selector,
		activePeerConnections: make(map[string]chan<- chan<- struct{}),
	}
}

type InputSource struct {
}

func NewSource() InputSource {
	return InputSource{}
}

func (source InputSource) Close() error {
	return nil
}

func (source InputSource) ID() string {
	generator, err := uuid.NewRandom()
	if err != nil {
		panic(err)
	}

	return generator.String()
}

type AudioTrack struct {
	*baseTrack
	*audio.Broadcaster
}

type VideoTrack struct {
	*baseTrack
	*video.Broadcaster
	shouldCopyFrames bool
}

const (
	rtpOutboundMTU = 1200
	rtcpInboundMTU = 1500
)

// Kind returns track's kind
func (track *baseTrack) Kind() webrtc.RTPCodecType {
	switch track.kind {
	case mediadevices.VideoInput:
		return webrtc.RTPCodecTypeVideo
	case mediadevices.AudioInput:
		return webrtc.RTPCodecTypeAudio
	default:
		panic("invalid track kind: only support VideoInput and AudioInput")
	}
}

func (track *baseTrack) StreamID() string {
	// TODO: StreamID should be used to group multiple tracks. Should get this information from mediastream instead.
	generator, err := uuid.NewRandom()
	if err != nil {
		panic(err)
	}

	return generator.String()
}

// RID is only relevant if you wish to use Simulcast
func (track *baseTrack) RID() string {
	return ""
}

// OnEnded sets an error handler. When a track has been created and started, if an
// error occurs, handler will get called with the error given to the parameter.
func (track *baseTrack) OnEnded(handler func(error)) {
	track.mu.Lock()
	track.onErrorHandler = handler
	err := track.err
	track.mu.Unlock()

	if err != nil && handler != nil {
		// Already errored.
		track.endOnce.Do(func() {
			handler(err)
		})
	}
}

// onError is a callback when an error occurs
func (track *baseTrack) onError(err error) {
	track.mu.Lock()
	track.err = err
	handler := track.onErrorHandler
	track.mu.Unlock()

	if handler != nil {
		track.endOnce.Do(func() {
			handler(err)
		})
	}
}

func (track *VideoTrack) Bind(ctx webrtc.TrackLocalContext) (webrtc.RTPCodecParameters, error) {
	return track.bind(ctx, track)
}

func (track *VideoTrack) Unbind(ctx webrtc.TrackLocalContext) error {
	return track.unbind(ctx)
}

func (track *AudioTrack) Bind(ctx webrtc.TrackLocalContext) (webrtc.RTPCodecParameters, error) {
	return track.bind(ctx, track)
}

func (track *AudioTrack) Unbind(ctx webrtc.TrackLocalContext) error {
	return track.unbind(ctx)
}

func (track *baseTrack) bind(ctx webrtc.TrackLocalContext, specializedTrack mediadevices.Track) (webrtc.RTPCodecParameters, error) {
	track.mu.Lock()
	defer track.mu.Unlock()

	signalCh := make(chan chan<- struct{})
	var stopRead chan struct{}
	track.activePeerConnections[ctx.ID()] = signalCh

	var encodedReader mediadevices.RTPReadCloser
	var selectedCodec webrtc.RTPCodecParameters
	var err error
	var errReasons []string
	for _, wantedCodec := range ctx.CodecParameters() {
		// logger.Debugf("trying to build %s rtp reader", wantedCodec.MimeType)
		encodedReader, err = specializedTrack.NewRTPReader(wantedCodec.MimeType, uint32(ctx.SSRC()), rtpOutboundMTU)
		if err == nil {
			selectedCodec = wantedCodec
			break
		}

		errReasons = append(errReasons, fmt.Sprintf("%s: %s", wantedCodec.MimeType, err))
	}

	if encodedReader == nil {
		return webrtc.RTPCodecParameters{}, errors.New(strings.Join(errReasons, "\n\n"))
	}

	go func() {
		var doneCh chan<- struct{}
		writer := ctx.WriteStream()
		defer func() {
			close(stopRead)
			encodedReader.Close()

			// When there's another call to unbind, it won't block since we remove the current ctx from active connections
			track.removeActivePeerConnection(ctx.ID())
			close(signalCh)
			if doneCh != nil {
				close(doneCh)
			}
		}()

		for {
			select {
			case doneCh = <-signalCh:
				return
			default:
			}

			pkts, _, err := encodedReader.Read()
			if err != nil {
				// explicitly ignore this error since the higher level should've reported this
				return
			}

			for _, pkt := range pkts {
				_, err = writer.WriteRTP(&pkt.Header, pkt.Payload)
				if err != nil {
					track.onError(err)
					return
				}
			}
		}
	}()

	keyFrameController, ok := encodedReader.Controller().(codec.KeyFrameController)
	if ok {
		stopRead = make(chan struct{})
		go track.rtcpReadLoop(ctx.RTCPReader(), keyFrameController, stopRead)
	}

	return selectedCodec, nil
}

func (track *baseTrack) rtcpReadLoop(reader interceptor.RTCPReader, keyFrameController codec.KeyFrameController, stopRead chan struct{}) {
	readerBuffer := make([]byte, rtcpInboundMTU)

readLoop:
	for {
		select {
		case <-stopRead:
			return
		default:
		}

		readLength, _, err := reader.Read(readerBuffer, interceptor.Attributes{})
		if err != nil {
			if errors.Is(err, io.EOF) {
				return
			}
			// logger.Warnf("failed to read rtcp packet: %s", err)
			continue
		}

		pkts, err := rtcp.Unmarshal(readerBuffer[:readLength])
		if err != nil {
			// logger.Warnf("failed to unmarshal rtcp packet: %s", err)
			continue
		}

		for _, pkt := range pkts {
			switch pkt.(type) {
			case *rtcp.PictureLossIndication, *rtcp.FullIntraRequest:
				if err := keyFrameController.ForceKeyFrame(); err != nil {
					// logger.Warnf("failed to force key frame: %s", err)
					continue readLoop
				}
			}
		}
	}
}

func (track *baseTrack) unbind(ctx webrtc.TrackLocalContext) error {
	ch := track.removeActivePeerConnection(ctx.ID())
	// If there isn't a registered chanel for this ctx, it means it has already been unbound
	if ch == nil {
		return nil
	}

	doneCh := make(chan struct{})
	ch <- doneCh
	<-doneCh
	return nil
}

func (track *baseTrack) removeActivePeerConnection(id string) chan<- chan<- struct{} {
	track.mu.Lock()
	defer track.mu.Unlock()

	ch, ok := track.activePeerConnections[id]
	if !ok {
		return nil
	}
	delete(track.activePeerConnections, id)

	return ch
}

func (track *AudioTrack) newEncodedReader(codecNames ...string) (mediadevices.EncodedReadCloser, *codec.RTPCodec, error) {
	reader := track.NewReader(false)
	inputProp, err := detectCurrentAudioProp(track.Broadcaster)
	if err != nil {
		return nil, nil, err
	}

	encodedReader, selectedCodec, err := track.selector.selectAudioCodecByNames(reader, inputProp, codecNames...)
	if err != nil {
		return nil, nil, err
	}

	sample := newAudioSampler(selectedCodec.ClockRate, selectedCodec.Latency)

	return &encodedReadCloserImpl{
		readFn: func() (mediadevices.EncodedBuffer, func(), error) {
			data, release, err := encodedReader.Read()
			buffer := mediadevices.EncodedBuffer{
				Data:    data,
				Samples: sample(),
			}
			return buffer, release, err
		},
		closeFn:      encodedReader.Close,
		controllerFn: encodedReader.Controller,
	}, selectedCodec, nil
}

func (track *AudioTrack) NewEncodedReader(codecName string) (mediadevices.EncodedReadCloser, error) {
	reader, _, err := track.newEncodedReader(codecName)
	return reader, err
}

func (track *AudioTrack) NewEncodedIOReader(codecName string) (io.ReadCloser, error) {
	encodedReader, _, err := track.newEncodedReader(codecName)
	if err != nil {
		return nil, err
	}
	return newEncodedIOReadCloserImpl(encodedReader), nil
}

func (track *AudioTrack) NewRTPReader(codecName string, ssrc uint32, mtu int) (mediadevices.RTPReadCloser, error) {
	encodedReader, selectedCodec, err := track.newEncodedReader(codecName)
	if err != nil {
		return nil, err
	}

	packetizer := rtp.NewPacketizer(uint16(mtu), uint8(selectedCodec.PayloadType), ssrc, selectedCodec.Payloader, rtp.NewRandomSequencer(), selectedCodec.ClockRate)

	return &rtpReadCloserImpl{
		readFn: func() ([]*rtp.Packet, func(), error) {
			encoded, release, err := encodedReader.Read()
			if err != nil {
				encodedReader.Close()
				track.onError(err)
				return nil, func() {}, err
			}
			defer release()

			pkts := packetizer.Packetize(encoded.Data, encoded.Samples)
			return pkts, release, err
		},
		closeFn:      encodedReader.Close,
		controllerFn: encodedReader.Controller,
	}, nil
}

func detectCurrentAudioProp(broadcaster *audio.Broadcaster) (prop.Media, error) {
	var currentProp prop.Media

	// Since broadcaster has a ring buffer internally, a new reader will either read the last
	// buffered frame or a new frame from the source. This also implies that no frame will be lost
	// in any case.
	metaReader := broadcaster.NewReader(false)
	metaReader = audio.DetectChanges(0, func(p prop.Media) { currentProp = p })(metaReader)
	_, _, err := metaReader.Read()

	return currentProp, err
}

func detectCurrentVideoProp(broadcaster *video.Broadcaster) (prop.Media, error) {
	var currentProp prop.Media

	// Since broadcaster has a ring buffer internally, a new reader will either read the last
	// buffered frame or a new frame from the source. This also implies that no frame will be lost
	// in any case.
	metaReader := broadcaster.NewReader(false)
	metaReader = video.DetectChanges(0, 0, func(p prop.Media) { currentProp = p })(metaReader)
	_, _, err := metaReader.Read()

	return currentProp, err
}

type samplerFunc func() uint32

func newAudioSampler(clockRate uint32, latency time.Duration) samplerFunc {
	samples := uint32(math.Round(float64(clockRate) * latency.Seconds()))
	return samplerFunc(func() uint32 {
		return samples
	})
}

// newVideoSampler creates a video sampler that uses the actual video frame rate and
// the codec's clock rate to come up with a duration for each sample.
func newVideoSampler(clockRate uint32) samplerFunc {
	clockRateFloat := float64(clockRate)
	lastTimestamp := time.Now()

	return samplerFunc(func() uint32 {
		now := time.Now()
		duration := now.Sub(lastTimestamp).Seconds()
		samples := uint32(math.Round(clockRateFloat * duration))
		lastTimestamp = now
		return samples
	})
}

// newFixedVideoSampler creates a video sampler that uses a fixed frame rate
// to ensure consistent timing regardless of actual frame arrival times.
func newFixedVideoSampler(clockRate uint32, framerate int) samplerFunc {
	samplesPerFrame := uint32(clockRate) / uint32(framerate)
	frameCount := 0
	cumulativeTimestamp := uint32(0)

	log.Printf("Fixed video sampler: clockRate=%d, framerate=%d, samplesPerFrame=%d", clockRate, framerate, samplesPerFrame)

	return samplerFunc(func() uint32 {
		frameCount++
		cumulativeTimestamp += samplesPerFrame
		if frameCount%30 == 1 {
			log.Printf("Frame %d: cumulative timestamp %d (increment %d)", frameCount, cumulativeTimestamp, samplesPerFrame)
		}
		return cumulativeTimestamp
	})
}

func (track *VideoTrack) newEncodedReader(codecNames ...string) (mediadevices.EncodedReadCloser, *codec.RTPCodec, error) {
	reader := track.NewReader(track.shouldCopyFrames)
	inputProp, err := detectCurrentVideoProp(track.Broadcaster)
	if err != nil {
		return nil, nil, err
	}

	encodedReader, selectedCodec, err := selectVideoCodecByNames(track.selector, reader, inputProp, codecNames...)
	if err != nil {
		return nil, nil, err
	}

	sample := newVideoSampler(selectedCodec.ClockRate)

	return &encodedReadCloserImpl{
		readFn: func() (mediadevices.EncodedBuffer, func(), error) {
			data, release, err := encodedReader.Read()
			buffer := mediadevices.EncodedBuffer{
				Data:    data,
				Samples: sample(),
			}
			return buffer, release, err
		},
		closeFn:      encodedReader.Close,
		controllerFn: encodedReader.Controller,
	}, selectedCodec, nil
}

func (track *VideoTrack) NewEncodedReader(codecName string) (mediadevices.EncodedReadCloser, error) {
	reader, _, err := track.newEncodedReader(codecName)
	return reader, err
}

func (track *VideoTrack) NewEncodedIOReader(codecName string) (io.ReadCloser, error) {
	panic("not implemented NewEncodedIOReader")
}

func (track *VideoTrack) NewRTPReader(codecName string, ssrc uint32, mtu int) (mediadevices.RTPReadCloser, error) {
	encodedReader, selectedCodec, err := track.newEncodedReader(codecName)
	if err != nil {
		return nil, err
	}

	packetizer := rtp.NewPacketizer(uint16(mtu), uint8(selectedCodec.PayloadType), ssrc, selectedCodec.Payloader, rtp.NewRandomSequencer(), selectedCodec.ClockRate)

	return &rtpReadCloserImpl{
		readFn: func() ([]*rtp.Packet, func(), error) {
			encoded, release, err := encodedReader.Read()
			if err != nil {
				encodedReader.Close()
				track.onError(err)
				return nil, func() {}, err
			}
			defer release()

			pkts := packetizer.Packetize(encoded.Data, encoded.Samples)
			return pkts, release, err
		},
		closeFn:      encodedReader.Close,
		controllerFn: encodedReader.Controller,
	}, nil
}

func newAudioTrackFromReader(reader audio.Reader, selector *CodecSelector) mediadevices.Track {
	base := newBaseTrack(mediadevices.AudioInput, selector)
	wrappedReader := audio.ReaderFunc(func() (chunk wave.Audio, release func(), err error) {
		chunk, _, err = reader.Read()
		if err != nil {
			// base.onError(err)
		}
		return chunk, func() {}, err
	})

	// TODO: Allow users to configure broadcaster
	broadcaster := audio.NewBroadcaster(wrappedReader, nil)

	return &AudioTrack{
		baseTrack:   base,
		Broadcaster: broadcaster,
	}
}

func newVideoTrackFromReader(reader video.Reader, selector *CodecSelector) mediadevices.Track {
	base := newBaseTrack(mediadevices.VideoInput, selector)
	wrappedReader := video.ReaderFunc(func() (img image.Image, release func(), err error) {
		img, _, err = reader.Read()
		if err != nil {
			// base.onError(err)
		}
		return img, func() {}, err
	})

	// TODO: Allow users to configure broadcaster
	broadcaster := video.NewBroadcaster(wrappedReader, nil)

	return &VideoTrack{
		baseTrack:   base,
		Broadcaster: broadcaster,
	}
}

// rtpreader.go

type rtpReadCloserImpl struct {
	readFn       func() ([]*rtp.Packet, func(), error)
	closeFn      func() error
	controllerFn func() codec.EncoderController
}

func (r *rtpReadCloserImpl) Read() ([]*rtp.Packet, func(), error) {
	return r.readFn()
}

func (r *rtpReadCloserImpl) Close() error {
	return r.closeFn()
}

func (r *rtpReadCloserImpl) Controller() codec.EncoderController {
	return r.controllerFn()
}

// ioreader.go

type encodedReadCloserImpl struct {
	readFn       func() (mediadevices.EncodedBuffer, func(), error)
	closeFn      func() error
	controllerFn func() codec.EncoderController
}

func (r *encodedReadCloserImpl) Read() (mediadevices.EncodedBuffer, func(), error) {
	return r.readFn()
}

func (r *encodedReadCloserImpl) Close() error {
	return r.closeFn()
}

func (r *encodedReadCloserImpl) Controller() codec.EncoderController {
	return r.controllerFn()
}

type encodedIOReadCloserImpl struct {
	readFn     func([]byte) (int, error)
	closeFn    func() error
	controller func() codec.EncoderController
}

func newEncodedIOReadCloserImpl(reader mediadevices.EncodedReadCloser) *encodedIOReadCloserImpl {
	var encoded mediadevices.EncodedBuffer
	release := func() {}
	return &encodedIOReadCloserImpl{
		readFn: func(b []byte) (int, error) {
			var err error

			if len(encoded.Data) == 0 {
				release()
				encoded, release, err = reader.Read()
				if err != nil {
					reader.Close()
					return 0, err
				}
			}

			n := copy(b, encoded.Data)
			encoded.Data = encoded.Data[n:]
			return n, nil
		},
		closeFn:    reader.Close,
		controller: reader.Controller,
	}
}

func (r *encodedIOReadCloserImpl) Read(b []byte) (int, error) {
	return r.readFn(b)
}

func (r *encodedIOReadCloserImpl) Close() error {
	return r.closeFn()
}

func (r *encodedIOReadCloserImpl) Controller() codec.EncoderController {
	return r.controller()
}

// codec.go

func selectVideoCodecByNames(selector *CodecSelector, reader video.Reader, inputProp prop.Media, codecNames ...string) (codec.ReadCloser, *codec.RTPCodec, error) {
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
