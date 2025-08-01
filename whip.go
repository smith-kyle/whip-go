package main

import (
	"bytes"
	"crypto/tls"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"

	"github.com/pion/mediadevices"
	"github.com/pion/webrtc/v3"
)

type WHIPClient struct {
	endpoint    string
	token       string
	resourceUrl string
}

func NewWHIPClient(endpoint string, token string) *WHIPClient {
	client := new(WHIPClient)
	client.endpoint = endpoint
	client.token = token
	return client
}

func (whip *WHIPClient) Publish(stream mediadevices.MediaStream, mediaEngine webrtc.MediaEngine, iceServers []webrtc.ICEServer, skipTlsAuth bool) {
	config := webrtc.Configuration{
		ICEServers: iceServers,
	}
	settings := webrtc.SettingEngine{}
	// settings.SetNetworkTypes([]webrtc.NetworkType{webrtc.NetworkTypeUDP4})

	pc, err := webrtc.NewAPI(
		webrtc.WithMediaEngine(&mediaEngine),
		webrtc.WithSettingEngine(settings),
	).NewPeerConnection(config)
	if err != nil {
		log.Fatal("Unexpected error building the PeerConnection. ", err)
	}

	for _, track := range stream.GetTracks() {
		track.OnEnded(func(err error) {
			log.Println("Track ended with error, ", err)
		})

		_, err = pc.AddTransceiverFromTrack(track,
			webrtc.RtpTransceiverInit{
				Direction: webrtc.RTPTransceiverDirectionSendonly,
			},
		)
		if err != nil {
			panic(err)
		}
	}

	pc.OnICEConnectionStateChange(func(connectionState webrtc.ICEConnectionState) {
		log.Printf("PeerConnection State has changed %s \n", connectionState.String())
	})

	offer, err := pc.CreateOffer(nil)
	if err != nil {
		log.Fatal("PeerConnection could not create offer. ", err)
	}
	err = pc.SetLocalDescription(offer)
	if err != nil {
		log.Fatal("PeerConnection could not set local offer. ", err)
	}

	// Block until ICE Gathering is complete, disabling trickle ICE
	// we do this because we only can exchange one signaling message
	// in a production application you should exchange ICE Candidates via OnICECandidate
	gatherComplete := webrtc.GatheringCompletePromise(pc)
	<-gatherComplete

	// log.Println(pc.LocalDescription().SDP)

	var sdp = []byte(pc.LocalDescription().SDP)
	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: skipTlsAuth,
			},
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			const MaxRedirectDepth = 10
			if len(via) >= MaxRedirectDepth {
				return http.ErrUseLastResponse
			}
			req.Header.Set("Authorization", "Bearer "+whip.token)
			return nil
		},
	}
	req, err := http.NewRequest("POST", whip.endpoint, bytes.NewBuffer(sdp))
	if err != nil {
		log.Fatal("Unexpected error building http request. ", err)
	}

	req.Header.Add("Content-Type", "application/sdp")
	if whip.token != "" {
		req.Header.Add("Authorization", "Bearer "+whip.token)
	}

	resp, err := client.Do(req)
	if err != nil {
		log.Fatal("Failed http POST request. ", err)
	}

	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)

	// Print the full SDP response
	log.Println("=== Server SDP Response ===")
	log.Println(string(body))
	log.Println("=== End SDP ===")

	// Parse and highlight interesting parts
	sdpLines := strings.Split(string(body), "\n")
	log.Println("\n=== Parsed SDP Info ===")
	for _, line := range sdpLines {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "c=IN IP4"):
			log.Printf("🌐 Connection Address: %s", line)
		case strings.HasPrefix(line, "a=candidate:"):
			log.Printf("🔗 ICE Candidate: %s", line)
		case strings.HasPrefix(line, "m="):
			log.Printf("📺 Media Line: %s", line)
		case strings.HasPrefix(line, "a=rtpmap:"):
			log.Printf("🎵 Codec Mapping: %s", line)
		case strings.HasPrefix(line, "a=fmtp:"):
			log.Printf("⚙️  Format Parameters: %s", line)
		}
	}
	log.Println("=== End Parsed Info ===")

	if resp.StatusCode != 201 {
		log.Fatalf("Non Successful POST: %d", resp.StatusCode)
	}

	resourceUrl, err := url.Parse(resp.Header.Get("Location"))
	if err != nil {
		log.Fatal("Failed to parse resource url. ", err)
	}
	base, err := url.Parse(whip.endpoint)
	if err != nil {
		log.Fatal("Failed to parse base url. ", err)
	}
	whip.resourceUrl = base.ResolveReference(resourceUrl).String()

	answer := webrtc.SessionDescription{}
	answer.Type = webrtc.SDPTypeAnswer
	answer.SDP = string(body)

	err = pc.SetRemoteDescription(answer)
	if err != nil {
		log.Fatal("PeerConnection could not set remote answer. ", err)
	}
}

func (whip *WHIPClient) Close(skipTlsAuth bool) {
	req, err := http.NewRequest("DELETE", whip.resourceUrl, nil)
	if err != nil {
		log.Fatal("Unexpected error building http request. ", err)
	}
	if whip.token != "" {
		req.Header.Add("Authorization", "Bearer "+whip.token)
	}

	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: skipTlsAuth,
			},
		},
	}
	_, err = client.Do(req)
	if err != nil {
		log.Fatal("Failed http DELETE request. ", err)
	}
}
