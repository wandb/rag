// Audio recording functionality
let recorder;

// Add WavStreamPlayer initialization and handling
let streamPlayer = null;

async function initializeStreamPlayer() {
    if (!streamPlayer) {
        streamPlayer = new WavStreamPlayer({ sampleRate: 44100 });
        await streamPlayer.connect();
    }
}

window.playAudioChunk = async function (element) {
    await initializeStreamPlayer();

    const base64Data = element.getAttribute('data-audio');
    const binaryString = window.atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);

    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Convert to Int16Array for PCM data
    const pcmData = new Int16Array(bytes.buffer);
    streamPlayer.add16BitPCM(pcmData);
};

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function () {
    // Listen for HTMX before swap events to handle recorder lifecycle
    htmx.on('htmx:beforeRequest', async function (evt) {
        // Check if this is a connect/disconnect request
        const path = evt.detail.pathInfo.requestPath;

        if (path === '/connect') {
            try {
                // Initialize recorder if it doesn't exist
                if (!recorder) {
                    // First check if we have microphone permission
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    console.log('Microphone permission granted:', stream);

                    // Create a temporary AudioContext to get the actual sample rate
                    const tempContext = new AudioContext();
                    const actualSampleRate = tempContext.sampleRate;
                    tempContext.close();

                    console.log('Detected device sample rate:', actualSampleRate);

                    recorder = new WavRecorder({
                        sampleRate: actualSampleRate,
                        outputToSpeakers: false
                    });

                    console.log('Recorder initialized:', recorder);
                    await recorder.begin();
                    console.log('Recorder ready for use');
                }
            } catch (error) {
                console.error('Recorder initialization error:', error);
                // Clean up if initialization failed
                if (recorder) {
                    try {
                        await recorder.quit();
                        recorder = null;
                    } catch (cleanupError) {
                        console.error('Error during cleanup:', cleanupError);
                    }
                }
                // Prevent the HTMX request if recorder initialization failed
                evt.preventDefault();
            }
        } else if (path === '/disconnect') {
            // Get the WebSocket container
            const wsContainer = document.getElementById('ws-container');
            if (wsContainer) {
                // Remove WebSocket attributes before disconnecting
                wsContainer.removeAttribute('hx-ext');
                wsContainer.removeAttribute('ws-connect');
            }

            if (recorder) {
                try {
                    // Only call quit() if the processor exists
                    if (recorder.processor) {
                        await recorder.quit();
                    }
                    recorder = null;
                    console.log('Recorder cleaned up successfully in beforeRequest');
                } catch (error) {
                    console.error('Error cleaning up recorder:', error);
                }
            }
        }
    });

    // Use event delegation for PTT button events
    document.addEventListener('mousedown', function (event) {
        const pttButton = event.target.closest('#ptt-btn');
        if (pttButton && !pttButton.disabled) {
            startRecording();
            // Change button appearance on mousedown
            pttButton.style.backgroundColor = '#ff4444';
            pttButton.style.color = 'white';
            pttButton.textContent = 'Release to Send';
        }
    });

    document.addEventListener('mouseup', function (event) {
        const pttButton = event.target.closest('#ptt-btn');
        if (pttButton && !pttButton.disabled) {
            stopRecording();
            // Revert button appearance on mouseup
            pttButton.style.backgroundColor = '';
            pttButton.style.color = '';
            pttButton.textContent = 'Push to Talk';
        }
    });


    // Add HTMX swap listener to debug button state after swaps
    htmx.on('htmx:afterSwap', function (evt) {
        const pttButton = document.getElementById('ptt-btn');
        if (pttButton) {
            console.log('Button swapped, new disabled state:', pttButton.disabled);
        }
        if (evt.detail.target.id === 'event-log') {
            evt.detail.target.scrollTop = evt.detail.target.scrollHeight;
        }
    });

    // Initialize HTMX WebSocket events
    htmx.on('htmx:wsConnecting', (evt) => {
        console.log('Connecting to WebSocket...', evt.detail.elt.id);
    });

    // Update WebSocket event listeners for better state tracking
    htmx.on('htmx:wsOpen', (evt) => {
        console.log('WebSocket Connected', evt.detail.elt.id);
    });

    // htmx.on('htmx:wsAfterMessage', (evt) => {
    //     console.log('WebSocket message received:', evt.detail.message);
    //     // Parse and handle the message if it's JSON
    //     try {
    //         const data = JSON.parse(evt.detail.message);
    //         if (data.type === 'connection_established') {
    //             console.log('Server confirmed connection:', data.message);
    //         }
    //     } catch (e) {
    //         // Handle non-JSON messages (like HTML updates)
    //         console.log('Received HTML update');
    // //     }
    // });

    htmx.on('audioReady', (evt) => {
        console.log('audioReady event received:', {
            target: evt.target.id,
            hasAudioPayload: !!evt.target.getAttribute('data-audio-payload')
        });
    });

    // Simplify to synchronous handler since no async operations remain
    htmx.on('htmx:wsConfigSend', (evt) => {
        console.log('htmx:wsConfigSend event received:', evt);

        const triggerEvent = evt.detail.triggeringEvent;
        if (triggerEvent && triggerEvent.type === 'audioMessage') {
            evt.detail.messageBody = JSON.stringify(triggerEvent.detail);
            console.log('Sending message:', evt.detail.messageBody.substring(0, 100) + '...');
        }
    });

    htmx.on('htmx:wsBeforeSend', (evt) => {
        console.log('Sending audio data from elt:', evt.detail.elt);
        console.log('Sending audio data:', evt.detail.message);

    });

    htmx.on('htmx:wsAfterSend', (evt) => {
        console.log('Audio data sent successfully from elt:', evt.detail.elt);
        console.log('Audio data sent successfully:', evt.detail.message);

    });

    htmx.on('htmx:wsClose', (evt) => {
        console.error('WebSocket connection closed', evt.detail);
    });

    htmx.on('htmx:wsError', (evt) => {
        console.error('WebSocket error:', evt.detail.error);
        console.error('Error event:', evt);
    });

    // Configure WebSocket behavior
    htmx.config.wsReconnectDelay = 'full-jitter';
    htmx.config.wsBinaryType = 'blob';

    // Simplify startRecording function since initialization is handled on connect
    async function startRecording() {
        try {
            if (!recorder) {
                console.error('Recorder not initialized');
                return;
            }

            // Start recording using the record() method
            await recorder.record();
            console.log('Recording started');
        } catch (error) {
            console.error('Start recording error:', error);
        }
    }

    async function stopRecording() {
        if (!recorder) {
            console.log('Recorder not initialized');
            return;
        }

        try {
            // End the recording and get the result
            const result = await recorder.end();

            // Handle file reading and data preparation
            const base64Data = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result.split(',')[1]);
                reader.onerror = reject;
                reader.readAsDataURL(result.blob);
            });

            // Send the audio data with metadata from the result
            const wsContainer = document.getElementById('ws-container');
            if (wsContainer) {
                htmx.trigger(wsContainer, 'audioMessage', {
                    type: 'audio',
                    data: base64Data,
                    metadata: {
                        sampleRate: result.sampleRate,
                        numberOfChannels: result.channelCount,
                        duration: result.duration,
                        format: 'audio/wav'
                    }
                });
            }

            // No need to clear since end() already cleaned up everything
            recorder = null;

        } catch (error) {
            console.error('Stop recording error:', error);
            // Cleanup on error
            if (recorder) {
                try {
                    await recorder.quit();
                    recorder = null;
                } catch (cleanupError) {
                    console.error('Error during cleanup:', cleanupError);
                }
            }
        }
    }
});