# Hanging

```
gst-launch-1.0 libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! v4l2h264enc extra-controls="controls,video_bitrate=4000000,h264_i_frame_period=60" ! 'video/x-h264,profile=constrained-baseline,level=(string)3.1' ! h264parse config-interval=1 ! rtph264pay pt=96 ! filesink location=output.h264 sync=false
```

# Fix hanging with v4l2convert

```
gst-launch-1.0 libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! v4l2convert ! queue ! v4l2h264enc extra-controls="controls,video_bitrate=4000000,h264_i_frame_period=60" ! 'video/x-h264,profile=constrained-baseline,level=(string)3.1' ! h264parse config-interval=1 ! rtph264pay pt=96 ! filesink location=output.h264 sync=false
```

# Based on hackaday but with level 3.1

```
gst-launch-1.0 libcamerasrc ! video/x-raw,width=640,height=480,format=NV12 ! v4l2convert ! queue ! v4l2h264enc ! 'video/x-h264,profile=constrained-baseline,level=(string)3.1' ! rtph264pay pt=96 mtu=1200 config-interval=-1 ! filesink location=output.h264 sync=false
```

# Stream suffix

```
! fdsink | ./whip-go -v h264-stream "$WHIP_URL"
```

# What claude recommended

```
gst-launch-1.0 libcamerasrc ! video/x-raw,width=640,height=480,format=NV12 ! v4l2convert ! queue ! v4l2h264enc extra-controls="controls,h264_profile=1,h264_level=9" ! 'video/x-h264,profile=constrained-baseline,level=(string)3.1' ! h264parse ! 'video/x-h264,stream-format=byte-stream,alignment=au' ! fdsink | ./whip-go -v h264-stream "$WHIP_URL"
```

# Attempt 1

```
gst-launch-1.0 libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! v4l2convert ! queue ! v4l2h264enc extra-controls="controls,video_bitrate=4000000,h264_i_frame_period=60" ! 'video/x-h264,profile=constrained-baseline,level=(string)3.1' ! h264parse config-interval=1 ! fdsink | ./whip-go -v h264-stream "$WHIP_URL"
```

# Produces a valid 42c01f .h264 file

```
gst-launch-1.0 libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! v4l2convert ! queue ! v4l2h264enc extra-controls="controls,video_bitrate=4000000,h264_i_frame_period=60" ! 'video/x-h264,profile=constrained-baseline,level=(string)3.1' ! h264parse config-interval=1 ! fdsink > test_stream.h264
```

```
gst-launch-1.0 libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! v4l2convert ! queue ! v4l2h264enc extra-controls="controls,video_bitrate=4000000,h264_i_frame_period=60" ! 'video/x-h264,profile=constrained-baseline,level=(string)3.1' ! h264parse config-interval=1 ! fdsink | ./whip-go -v h264-stream "$WHIP_URL
```
