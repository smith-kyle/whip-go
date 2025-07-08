package main

import (
	"bytes"
	"crypto/tls"
	"io"
	"log"
	"net/http"
	"net/url"

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
		log.Printf("ICE Connection State changed: %s", connectionState.String())

		switch connectionState {
		case webrtc.ICEConnectionStateFailed:
			log.Printf("ICE connection failed - this usually means network connectivity issues")
		case webrtc.ICEConnectionStateDisconnected:
			log.Printf("ICE connection disconnected")
		case webrtc.ICEConnectionStateConnected:
			log.Printf("ICE connection established successfully!")
		case webrtc.ICEConnectionStateCompleted:
			log.Printf("ICE connection completed successfully!")
		}
	})

	pc.OnConnectionStateChange(func(connectionState webrtc.PeerConnectionState) {
		log.Printf("PeerConnection State changed: %s", connectionState.String())
	})

	pc.OnICECandidate(func(candidate *webrtc.ICECandidate) {
		if candidate != nil {
			log.Printf("ICE Candidate: %s %s %d", candidate.Protocol, candidate.Address, candidate.Port)
		}
	})

	pc.OnICEGatheringStateChange(func(state webrtc.ICEGathererState) {
		log.Printf("ICE Gathering State changed: %s", state.String())
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

	localSDP := pc.LocalDescription().SDP
	log.Printf("Local SDP Offer length: %d bytes", len(localSDP))

	// Show first 200 characters of SDP for debugging
	sdpPreview := localSDP
	if len(sdpPreview) > 200 {
		sdpPreview = sdpPreview[:200]
	}
	log.Printf("Local SDP Offer (first 200 chars): %s", sdpPreview)

	var sdp = []byte(localSDP)
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
	log.Printf("Sending WHIP POST request to: %s", whip.endpoint)
	req, err := http.NewRequest("POST", whip.endpoint, bytes.NewBuffer(sdp))
	if err != nil {
		log.Fatal("Unexpected error building http request. ", err)
	}

	req.Header.Add("Content-Type", "application/sdp")
	if whip.token != "" {
		req.Header.Add("Authorization", "Bearer "+whip.token)
		log.Printf("Using authentication token")
	}

	log.Printf("Sending WHIP request with %d bytes of SDP", len(sdp))
	resp, err := client.Do(req)
	if err != nil {
		log.Fatal("Failed http POST request. ", err)
	}

	log.Printf("WHIP server response: %d %s", resp.StatusCode, resp.Status)

	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatal("Failed to read response body. ", err)
	}

	log.Printf("Response body length: %d bytes", len(body))
	log.Printf("Response headers: %v", resp.Header)

	if resp.StatusCode != 201 {
		log.Printf("Response body: %s", string(body))
		log.Fatalf("Non Successful POST: %d - %s", resp.StatusCode, string(body))
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
	log.Printf("Resource URL: %s", whip.resourceUrl)

	answer := webrtc.SessionDescription{}
	answer.Type = webrtc.SDPTypeAnswer
	answer.SDP = string(body)

	// Show first 200 characters of SDP answer for debugging
	answerPreview := answer.SDP
	if len(answerPreview) > 200 {
		answerPreview = answerPreview[:200]
	}
	log.Printf("Remote SDP Answer (first 200 chars): %s", answerPreview)

	log.Printf("Setting remote description...")
	err = pc.SetRemoteDescription(answer)
	if err != nil {
		log.Fatal("PeerConnection could not set remote answer. ", err)
	}
	log.Printf("Remote description set successfully")
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
