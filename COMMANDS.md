# Run with 15 fps

```
gst-launch-1.0 libcamerasrc ! video/x-raw,width=640,height=360,framerate=15/1 ! v4l2convert ! queue ! v4l2h264enc extra-controls="controls,video_bitrate=800000,h264_i_frame_period=30" !  'video/x-h264,profile=constrained-baseline,level=(string)3.1' ! h264parse config-interval=1 !  fdsink | ./whip-go -v h264-stream -fps 15 "$WHIP_URL"
```
