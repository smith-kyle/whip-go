package main

import (
	"bytes"
	"errors"
	"io"
	"log"
	"sync"
	"time"

	"github.com/pion/mediadevices"
	"github.com/pion/mediadevices/pkg/codec"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v3"
)

// H.264 NAL unit types
const (
	NALUnitTypeSlice = 1
	NALUnitTypeIDR   = 5
	NALUnitTypeSPS   = 7
	NALUnitTypePPS   = 8
	NALUnitTypeAUD   = 9
	NALUnitTypeSEI   = 6
)

// H.264 NAL unit structure
type H264NALUnit struct {
	Type       uint8
	Data       []byte
	Timestamp  time.Time
	IsKeyFrame bool
}

// H.264 stream reader for parsing NAL units
type H264StreamReader struct {
	reader io.ReadCloser
	buffer []byte
	mu     sync.Mutex
}

// NewH264StreamReader creates a new H.264 stream reader
func NewH264StreamReader(reader io.ReadCloser) *H264StreamReader {
	return &H264StreamReader{
		reader: reader,
		buffer: make([]byte, 0, 1024*1024), // 1MB buffer
	}
}

// ReadNALUnit reads the next NAL unit from the stream
func (r *H264StreamReader) ReadNALUnit() (*H264NALUnit, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Look for start codes (0x00000001 or 0x000001)
	for {
		// Read more data if buffer is too small
		if len(r.buffer) < 4 {
			tmp := make([]byte, 4096)
			n, err := r.reader.Read(tmp)
			if err != nil {
				return nil, err
			}
			r.buffer = append(r.buffer, tmp[:n]...)
		}

		// Find start code
		startCodeLen := r.findStartCode()
		if startCodeLen == 0 {
			// No start code found, read more data
			tmp := make([]byte, 4096)
			n, err := r.reader.Read(tmp)
			if err != nil {
				return nil, err
			}
			r.buffer = append(r.buffer, tmp[:n]...)
			continue
		}

		// Skip the start code
		r.buffer = r.buffer[startCodeLen:]

		// Find the next start code to determine NAL unit length
		nextStartCodePos := r.findNextStartCode()
		if nextStartCodePos == -1 {
			// No next start code found, read more data
			tmp := make([]byte, 4096)
			n, err := r.reader.Read(tmp)
			if err != nil && err != io.EOF {
				return nil, err
			}
			if n > 0 {
				r.buffer = append(r.buffer, tmp[:n]...)
				continue
			}
			// EOF reached, use remaining buffer
			nextStartCodePos = len(r.buffer)
		}

		// Extract NAL unit
		nalData := make([]byte, nextStartCodePos)
		copy(nalData, r.buffer[:nextStartCodePos])
		r.buffer = r.buffer[nextStartCodePos:]

		if len(nalData) == 0 {
			continue
		}

		// Parse NAL unit header
		nalType := nalData[0] & 0x1F
		isKeyFrame := nalType == NALUnitTypeIDR || nalType == NALUnitTypeSPS || nalType == NALUnitTypePPS

		return &H264NALUnit{
			Type:       nalType,
			Data:       nalData,
			Timestamp:  time.Now(),
			IsKeyFrame: isKeyFrame,
		}, nil
	}
}

// findStartCode finds the start code at the beginning of the buffer
func (r *H264StreamReader) findStartCode() int {
	if len(r.buffer) < 3 {
		return 0
	}

	// Check for 4-byte start code (0x00000001)
	if len(r.buffer) >= 4 && bytes.Equal(r.buffer[:4], []byte{0x00, 0x00, 0x00, 0x01}) {
		return 4
	}

	// Check for 3-byte start code (0x000001)
	if bytes.Equal(r.buffer[:3], []byte{0x00, 0x00, 0x01}) {
		return 3
	}

	return 0
}

// findNextStartCode finds the position of the next start code
func (r *H264StreamReader) findNextStartCode() int {
	for i := 1; i < len(r.buffer)-2; i++ {
		// Check for 4-byte start code
		if i < len(r.buffer)-3 && bytes.Equal(r.buffer[i:i+4], []byte{0x00, 0x00, 0x00, 0x01}) {
			return i
		}
		// Check for 3-byte start code
		if bytes.Equal(r.buffer[i:i+3], []byte{0x00, 0x00, 0x01}) {
			return i
		}
	}
	return -1
}

// Close closes the stream reader
func (r *H264StreamReader) Close() error {
	return r.reader.Close()
}

// H.264 stream track implementation
type H264StreamTrack struct {
	*baseTrack
	reader      *H264StreamReader
	clockRate   uint32
	payloadType uint8
	ssrc        uint32
}

// NewH264StreamTrack creates a new H.264 stream track
func NewH264StreamTrack(reader io.ReadCloser, selector *CodecSelector) mediadevices.Track {
	base := newBaseTrack(mediadevices.VideoInput, selector)

	return &H264StreamTrack{
		baseTrack:   base,
		reader:      NewH264StreamReader(reader),
		clockRate:   90000, // H.264 standard clock rate
		payloadType: 96,    // Dynamic payload type for H.264
	}
}

// Bind binds the track to a WebRTC context
func (track *H264StreamTrack) Bind(ctx webrtc.TrackLocalContext) (webrtc.RTPCodecParameters, error) {
	return track.bind(ctx, track)
}

// Unbind unbinds the track from a WebRTC context
func (track *H264StreamTrack) Unbind(ctx webrtc.TrackLocalContext) error {
	return track.unbind(ctx)
}

// NewRTPReader creates an RTP reader for H.264 stream
func (track *H264StreamTrack) NewRTPReader(codecName string, ssrc uint32, mtu int) (mediadevices.RTPReadCloser, error) {
	track.ssrc = ssrc

	// Create H.264 RTP packetizer
	packetizer := rtp.NewPacketizer(
		uint16(mtu),
		track.payloadType,
		ssrc,
		&H264Payloader{},
		rtp.NewRandomSequencer(),
		track.clockRate,
	)

	return &rtpReadCloserImpl{
		readFn: func() ([]*rtp.Packet, func(), error) {
			nalUnit, err := track.reader.ReadNALUnit()
			if err != nil {
				if err.Error() != "EOF" {
					log.Printf("H.264 NAL unit read error: %v", err)
				}
				track.onError(err)
				return nil, func() {}, err
			}

			// Calculate timestamp (90kHz clock)
			timestamp := uint32(nalUnit.Timestamp.UnixNano() / 1000000 * 90 / 1000)

			log.Printf("H.264 NAL unit: type=%d, size=%d bytes, keyframe=%v",
				nalUnit.Type, len(nalUnit.Data), nalUnit.IsKeyFrame)

			// Packetize NAL unit
			packets := packetizer.Packetize(nalUnit.Data, timestamp)

			log.Printf("Packetized into %d RTP packets", len(packets))

			return packets, func() {}, nil
		},
		closeFn: func() error {
			return track.reader.Close()
		},
		controllerFn: func() codec.EncoderController {
			return &H264StreamController{track: track}
		},
	}, nil
}

// NewEncodedReader creates an encoded reader (not implemented for H.264 stream)
func (track *H264StreamTrack) NewEncodedReader(codecName string) (mediadevices.EncodedReadCloser, error) {
	return nil, errors.New("NewEncodedReader not supported for H.264 stream track")
}

// NewEncodedIOReader creates an encoded IO reader (not implemented for H.264 stream)
func (track *H264StreamTrack) NewEncodedIOReader(codecName string) (io.ReadCloser, error) {
	return nil, errors.New("NewEncodedIOReader not supported for H.264 stream track")
}

// H.264 RTP Payloader implementation
type H264Payloader struct{}

func (p *H264Payloader) Payload(mtu uint16, payload []byte) [][]byte {
	if len(payload) == 0 {
		return [][]byte{}
	}

	maxPayloadSize := int(mtu) - 12 // RTP header size

	// Single NAL unit mode
	if len(payload) <= maxPayloadSize {
		return [][]byte{payload}
	}

	// Fragmentation Unit (FU-A) mode
	var payloads [][]byte
	nalType := payload[0] & 0x1F
	nalRef := payload[0] & 0x60

	// FU indicator: F=0, NRI=from original, Type=28 (FU-A)
	fuIndicator := nalRef | 28

	// Skip NAL header for fragmentation
	payload = payload[1:]

	for len(payload) > 0 {
		payloadSize := len(payload)
		if payloadSize > maxPayloadSize-2 { // Reserve space for FU indicator and header
			payloadSize = maxPayloadSize - 2
		}

		// FU header: S, E, R, Type
		fuHeader := nalType
		if len(payloads) == 0 {
			fuHeader |= 0x80 // Start bit
		}
		if payloadSize == len(payload) {
			fuHeader |= 0x40 // End bit
		}

		// Create fragmented payload
		fragPayload := make([]byte, 2+payloadSize)
		fragPayload[0] = fuIndicator
		fragPayload[1] = fuHeader
		copy(fragPayload[2:], payload[:payloadSize])

		payloads = append(payloads, fragPayload)
		payload = payload[payloadSize:]
	}

	return payloads
}

// H.264 Stream Controller for key frame requests
type H264StreamController struct {
	track *H264StreamTrack
}

func (c *H264StreamController) ForceKeyFrame() error {
	// For H.264 streams, we can't force keyframes as we're just passing through
	// This would need to be implemented at the source (libcamera-vid)
	return nil
}

// GetInputH264Stream creates a media stream from H.264 input
func GetInputH264Stream(reader io.ReadCloser, codecSelector *CodecSelector) (mediadevices.MediaStream, error) {
	track := NewH264StreamTrack(reader, codecSelector)

	stream, err := mediadevices.NewMediaStream(track)
	if err != nil {
		return nil, err
	}

	return stream, nil
}
