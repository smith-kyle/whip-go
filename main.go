package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/pion/mediadevices"
	"github.com/pion/mediadevices/pkg/codec/opus"
	"github.com/pion/mediadevices/pkg/codec/vpx"
	"github.com/pion/mediadevices/pkg/codec/x264"
	// _ "github.com/pion/mediadevices/pkg/driver/screen" // This is required to register screen adapter
	"github.com/pion/mediadevices/pkg/prop"

	//_ "github.com/pion/mediadevices/pkg/driver/camera"
	//_ "github.com/pion/mediadevices/pkg/driver/microphone"
	_ "github.com/pion/mediadevices/pkg/driver/audiotest"
	_ "github.com/pion/mediadevices/pkg/driver/videotest"
	"github.com/pion/webrtc/v3"
)

func main() {
	video := flag.String("v", "screen", "input video device, can be \"screen\", \"h264-stream\", or a named pipe")
	audio := flag.String("a", "", "input audio device, can be a named pipe")
	videoBitrate := flag.Int("b", 1_000_000, "video bitrate in bits per second")
	iceServer := flag.String("i", "stun:stun.l.google.com:19302", "ice server")
	token := flag.String("t", "", "publishing token")
	videoCodec := flag.String("vc", "vp8", "video codec vp8|h264")
	framerate := flag.Int("fps", 30, "framerate for h264-stream mode")
	flag.Parse()

	if len(flag.Args()) != 1 {
		log.Fatal("Invalid number of arguments, pass the publishing url as the first argument")
	}

	mediaEngine := webrtc.MediaEngine{}
	whip := NewWHIPClient(flag.Args()[0], *token)

	// configure codec specific parameters
	vpxParams, err := vpx.NewVP8Params()
	if err != nil {
		panic(err)
	}
	vpxParams.BitRate = *videoBitrate

	opusParams, err := opus.NewParams()
	if err != nil {
		panic(err)
	}

	x264Params, err := x264.NewParams()
	if err != nil {
		panic(err)
	}
	x264Params.BitRate = *videoBitrate
	x264Params.Preset = x264.PresetUltrafast

	var videoCodecSelector mediadevices.CodecSelectorOption
	if *videoCodec == "vp8" {
		videoCodecSelector = mediadevices.WithVideoEncoders(&vpxParams)
	} else {
		videoCodecSelector = mediadevices.WithVideoEncoders(&x264Params)
	}
	var stream mediadevices.MediaStream

	if *video == "screen" {
		codecSelector := mediadevices.NewCodecSelector(videoCodecSelector)
		codecSelector.Populate(&mediaEngine)

		stream, err = mediadevices.GetDisplayMedia(mediadevices.MediaStreamConstraints{
			Video: func(constraint *mediadevices.MediaTrackConstraints) {},
			Codec: codecSelector,
		})
		if err != nil {
			log.Fatal("Unexpected error capturing screen. ", err)
		}
	} else if *video == "test" {
		codecSelector := mediadevices.NewCodecSelector(
			videoCodecSelector,
			mediadevices.WithAudioEncoders(&opusParams),
		)
		codecSelector.Populate(&mediaEngine)

		stream, err = mediadevices.GetUserMedia(mediadevices.MediaStreamConstraints{
			Video: func(constraint *mediadevices.MediaTrackConstraints) {
				constraint.Width = prop.Int(640)
				constraint.Height = prop.Int(480)
			},
			Audio: func(constraint *mediadevices.MediaTrackConstraints) {},
			Codec: codecSelector,
		})
		if err != nil {
			log.Fatal("Unexpected error capturing test source. ", err)
		}
	} else if *video == "h264-stream" {
		codecSelector := NewCodecSelector(
			WithVideoEncoders(&x264Params),
		)
		codecSelector.Populate(&mediaEngine)

		stream, err = GetH264StreamFromStdin(codecSelector, *framerate)
		if err != nil {
			log.Fatal("Unexpected error capturing H.264 stream from stdin. ", err)
		}
	} else {
		codecSelector := NewCodecSelector(
			WithVideoEncoders(&vpxParams),
			WithAudioEncoders(&opusParams),
		)
		codecSelector.Populate(&mediaEngine)

		stream, err = GetInputMediaStream(*audio, *video, codecSelector)
		if err != nil {
			log.Fatal("Unexpected error capturing input pipe. ", err)
		}
	}

	iceServers := []webrtc.ICEServer{
		{
			URLs: []string{*iceServer},
		},
	}

	whip.Publish(stream, mediaEngine, iceServers, true)

	if *video == "h264-stream" {
		// For h264-stream, don't read from stdin (it's being used for video data)
		// Wait for interrupt signal instead
		fmt.Println("Streaming H.264 from stdin. Press Ctrl+C to stop...")
		select {} // Block forever until interrupted
	} else {
		fmt.Println("Press 'Enter' to finish...")
		bufio.NewReader(os.Stdin).ReadBytes('\n')
	}

	whip.Close(true)
}
