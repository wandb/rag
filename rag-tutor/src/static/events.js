// Audio recording functionality
let recorder;

// Add state tracking
let isRecording = false;
let isProcessing = false;

// Add visualization intervals tracking
const visualizationIntervals = new Map();

// Recording functions
async function startRecording() {
    try {
        // Prevent multiple simultaneous operations
        if (isProcessing || isRecording) {
            return;
        }

        isProcessing = true;

        // Only send cancel event if there are existing audio chunks
        const wsContainer = document.getElementById('ws-container');
        if (wsContainer && audioChunks.length > 0) {
            const eventDetail = {
                type: 'cancel',
                data: 'Cancel current audio stream'
            };

            const audioMessageEvent = new CustomEvent('audioMessage', { bubbles: true, detail: eventDetail });
            wsContainer.dispatchEvent(audioMessageEvent);
        }

        // Reset audio player before starting new recording
        if (streamPlayer) {
            await streamPlayer.reset();
        }
        audioChunks = [];
        isFirstChunk = true;

        // Reinitialize recorder if needed
        if (!recorder) {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const tempContext = new AudioContext();
            const actualSampleRate = tempContext.sampleRate;
            tempContext.close();

            recorder = new WavRecorder({
                sampleRate: actualSampleRate,
                outputToSpeakers: false
            });
            await recorder.begin();
        }

        await recorder.record();
        isRecording = true;
        // console.log('Recording started');
    } catch (error) {
        console.error('Start recording error:', error);
        // Clean up on error
        if (recorder) {
            try {
                await recorder.quit();
                recorder = null;
            } catch (cleanupError) {
                console.error('Error during cleanup:', cleanupError);
            }
        }
    } finally {
        isProcessing = false;
    }
}

async function stopRecording() {
    if (!isRecording || isProcessing) {
        // console.log('No active recording or processing in progress');
        return;
    }

    try {
        isProcessing = true;

        if (!recorder) {
            // console.log('Recorder not initialized');
            return;
        }

        const result = await recorder.end();
        isRecording = false;

        const base64Data = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                resolve(reader.result.split(',')[1]);
            };
            reader.onerror = (error) => {
                console.error('FileReader error:', error);
                reject(error);
            };
            reader.readAsDataURL(result.blob);
        });

        const wsContainer = document.getElementById('ws-container');
        if (wsContainer) {
            const eventDetail = {
                type: 'audio',
                data: base64Data,
                metadata: {
                    sampleRate: result.sampleRate,
                    numberOfChannels: result.channelCount,
                    duration: result.duration,
                    format: 'audio/wav'
                }
            };

            const audioMessageEvent = new CustomEvent('audioMessage', { bubbles: true, detail: eventDetail });
            wsContainer.dispatchEvent(audioMessageEvent);
        }

        // Reinitialize recorder
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const tempContext = new AudioContext();
            const actualSampleRate = tempContext.sampleRate;
            tempContext.close();

            recorder = new WavRecorder({
                sampleRate: actualSampleRate,
                outputToSpeakers: false
            });
            await recorder.begin();
        } catch (initError) {
            console.error('Failed to reinitialize recorder:', initError);
            recorder = null;
        }

    } catch (error) {
        console.error('Stop recording error:', error);
        if (recorder) {
            try {
                await recorder.quit();
                recorder = null;
            } catch (cleanupError) {
                console.error('Error during cleanup:', cleanupError);
            }
        }
    } finally {
        isProcessing = false;
    }
}

// Audio player functionality
let streamPlayer;
let audioElement;
let audioBuffer;
let isPlaying = false;
let audioChunks = [];
let isFirstChunk = true;

// Update the sample rate constant
const OPENAI_SAMPLE_RATE = 24000;  // OpenAI uses 24kHz



async function initializeAudioPlayer(autoplay = true) {
    try {
        audioElement = document.getElementById('audio-player');
        if (!audioElement) {
            console.error('Audio element not found');
            return;
        }

        // Initialize WavStreamPlayer with correct sample rate
        streamPlayer = new WavStreamPlayer({
            sampleRate: OPENAI_SAMPLE_RATE
        });
        await streamPlayer.connect(audioElement);

        // Set audio element properties
        audioElement.autoplay = true;
        audioElement.muted = false;

        // Try to enable autoplay
        if (autoplay) {
            try {
                await audioElement.play();
            } catch (e) {
                console.log('Initial autoplay failed, waiting for user interaction');
                const interactionEvents = ['click', 'touchstart', 'keydown'];
                const playHandler = async () => {
                    try {
                        await streamPlayer.context.resume();
                        await audioElement.play();
                        // Remove all event listeners once played
                        interactionEvents.forEach(event =>
                            document.removeEventListener(event, playHandler));
                    } catch (err) {
                        console.error('Play after interaction failed:', err);
                    }
                };

                interactionEvents.forEach(event =>
                    document.addEventListener(event, playHandler));
            }
        }

        return true;
    } catch (error) {
        console.error('Error initializing audio player:', error);
        return false;
    }
}

// Function to process incoming audio chunks
let updateTimeout = null;
const UPDATE_DELAY = 1000; // Update every second

async function processAudioChunk(base64Data) {
    try {
        if (!streamPlayer) {
            await initializeAudioPlayer(true);
        }

        // For the first chunk, reset everything
        if (isFirstChunk) {
            audioChunks = []; // Clear existing chunks
            if (streamPlayer) {
                await streamPlayer.reset();
            }
            isPlaying = true;
            isFirstChunk = false;
            // Immediately process first chunk without delay
            updateAudioSource(true);
        }

        // Convert base64 to binary data
        const binaryData = atob(base64Data);
        const arrayBuffer = new ArrayBuffer(binaryData.length);
        const uint8Array = new Uint8Array(arrayBuffer);
        for (let i = 0; i < binaryData.length; i++) {
            uint8Array[i] = binaryData.charCodeAt(i);
        }

        // Store the chunk
        audioChunks.push(uint8Array);

        // Clear existing timeout
        if (updateTimeout) {
            clearTimeout(updateTimeout);
        }

        // Process immediately for the last chunk (when it's small)
        if (uint8Array.length < 1024) {  // Assuming small chunks are final chunks
            updateAudioSource(false);
        } else {
            // For larger chunks, debounce the updates
            updateTimeout = setTimeout(() => updateAudioSource(false), UPDATE_DELAY);
        }

    } catch (error) {
        console.error('Error processing audio chunk:', error);
    }
}

function updateAudioSource(isFirst) {
    if (!audioElement) return;

    const currentTime = audioElement.currentTime;
    const wasPlaying = !audioElement.paused;

    // Create WAV header and concatenate all chunks
    const wavHeader = createWavHeader(audioChunks);
    const chunks = [wavHeader, ...audioChunks];

    const audioBlob = new Blob(chunks, { type: 'audio/wav' });

    // Clean up old audio URL
    if (audioElement.src) {
        URL.revokeObjectURL(audioElement.src);
    }

    const audioUrl = URL.createObjectURL(audioBlob);
    audioElement.src = audioUrl;

    // Restore playback state
    if (!isFirst) {
        audioElement.currentTime = currentTime;
    }

    // Always try to play for first chunk or if it was already playing
    if (isFirst || wasPlaying) {
        const playPromise = audioElement.play();
        if (playPromise !== undefined) {
            playPromise.catch(error => {
                console.log('Autoplay failed:', error);
                // Add one-time event listener for user interaction
                const playHandler = () => {
                    audioElement.play()
                        .catch(e => console.error('Delayed autoplay failed:', e));
                    document.removeEventListener('click', playHandler);
                };
                document.addEventListener('click', playHandler);
            });
        }
    }
}

// Helper function to create WAV header
function createWavHeader(chunks) {
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const header = new ArrayBuffer(44);
    const view = new DataView(header);

    // "RIFF" chunk descriptor
    view.setUint32(0, 0x52494646, false); // "RIFF"
    view.setUint32(4, 36 + totalLength, true); // File size
    view.setUint32(8, 0x57415645, false); // "WAVE"

    // "fmt " sub-chunk
    view.setUint32(12, 0x666D7420, false); // "fmt "
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true); // NumChannels (1 for mono)
    view.setUint32(24, OPENAI_SAMPLE_RATE, true); // SampleRate (24kHz)
    view.setUint32(28, OPENAI_SAMPLE_RATE * 2, true); // ByteRate (sampleRate * blockAlign)
    view.setUint16(32, 2, true); // BlockAlign (channels * bitsPerSample/8)
    view.setUint16(34, 16, true); // BitsPerSample

    // "data" sub-chunk
    view.setUint32(36, 0x64617461, false); // "data"
    view.setUint32(40, totalLength, true); // Subchunk2Size

    return new Uint8Array(header);
}

// Configure WebSocket behavior
htmx.config.wsReconnectDelay = 'full-jitter';
htmx.config.wsBinaryType = 'blob';

// WebSocket Connection Lifecycle Events
htmx.on('htmx:wsConnecting', (evt) => {
    // console.log('Connecting to WebSocket...', evt.detail.elt.id);
});

htmx.on('htmx:wsOpen', async (evt) => {
    // console.log('WebSocket Connected', evt.detail.elt.id);
    await initializeAudioPlayer(false);
});

// Clean up when WebSocket closes
htmx.on('htmx:wsClose', async (evt) => {
    // console.log('WebSocket connection closed');
    audioChunks = []; // Clear stored chunks

    try {
        if (streamPlayer) {
            await streamPlayer.interrupt(); // This will stop playback
            streamPlayer = null;
        }

        if (audioElement) {
            audioElement.pause();
            audioElement.src = '';
            audioElement.load();
        }
    } catch (error) {
        console.error('Error cleaning up WebSocket resources:', error);
    }
});

htmx.on('htmx:wsError', (evt) => {
    console.error('WebSocket error:', evt.detail.error);
});

// WebSocket Message Handling Events
htmx.on('htmx:wsConfigSend', (evt) => {
    const triggerEvent = evt.detail.triggeringEvent;
    if (triggerEvent && triggerEvent.type === 'audioMessage') {
        evt.detail.messageBody = JSON.stringify({
            type: 'audio',
            ...triggerEvent.detail
        });
        // console.log('Configured WebSocket message:', evt.detail.messageBody.substring(0, 100) + '...');
    }
});

htmx.on('htmx:wsBeforeSend', (evt) => {
    // console.log('Sending audio data from elt:', evt.detail.elt);
});

htmx.on('htmx:wsAfterSend', (evt) => {
    // console.log('Audio data sent successfully from elt:', evt.detail.elt);
});

htmx.on('htmx:wsBeforeMessage', async function (evt) {
    const message = evt.detail.message;

    try {
        // Try to parse the message if it's a string
        const parsedMessage = typeof message === 'string' ? JSON.parse(message) : message;

        // Handle audio messages
        if (parsedMessage.type === 'audio') {
            evt.preventDefault(); // Prevent default HTMX processing
            await processAudioChunk(parsedMessage.data);
            return;
        }
    } catch (error) {
        // If parsing fails, it's not JSON data, let HTMX handle it silently
    }

    // Let HTMX handle non-audio messages for DOM updates
});

// Other HTMX Events
htmx.on('htmx:beforeRequest', async function (evt) {
    const path = evt.detail.pathInfo.requestPath;
    if (path === '/connect') {
        try {
            if (!recorder) {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const tempContext = new AudioContext();
                const actualSampleRate = tempContext.sampleRate;
                tempContext.close();

                recorder = new WavRecorder({
                    sampleRate: actualSampleRate,
                    outputToSpeakers: false
                });
                await recorder.begin();
            }
        } catch (error) {
            console.error('Recorder initialization error:', error);
            if (recorder) {
                try {
                    await recorder.quit();
                    recorder = null;
                } catch (cleanupError) {
                    console.error('Error during cleanup:', cleanupError);
                }
            }
            evt.preventDefault();
        }
    } else if (path === '/disconnect') {
        audioChunks = [];
        if (audioElement) {
            audioElement.removeAttribute('src'); // Remove src instead of setting to empty string
            audioElement.load(); // Force reload
        }
        // Clean up all resources
        if (recorder) {
            try {
                if (recorder.processor) {
                    await recorder.quit();
                }
                recorder = null;
            } catch (error) {
                console.error('Error cleaning up recorder:', error);
            }
        }

        if (streamPlayer) {
            await streamPlayer.interrupt();
            streamPlayer = null;
        }

        for (const interval of visualizationIntervals.values()) {
            clearInterval(interval);
        }
        visualizationIntervals.clear();
    }
});

htmx.on('htmx:afterSwap', function (evt) {
    const pttButton = document.getElementById('ptt-btn');
    if (pttButton) {
        // console.log('Button swapped, new disabled state:', pttButton.disabled);
    }
    if (evt.detail.target.id === 'event-log') {
        evt.detail.target.scrollTop = evt.detail.target.scrollHeight;
    }
});

// DOM event listeners
document.addEventListener('DOMContentLoaded', function () {
    // PTT button event listeners
    document.addEventListener('mousedown', function (event) {
        const pttButton = event.target.closest('#ptt-btn');
        if (pttButton && !pttButton.disabled && !isProcessing) {
            startRecording();
            pttButton.style.backgroundColor = '#ff4444';
            pttButton.style.color = 'white';
            pttButton.textContent = 'Release to Send';
        }
    });

    document.addEventListener('mouseup', function (event) {
        const pttButton = event.target.closest('#ptt-btn');
        if (pttButton && !pttButton.disabled && isRecording) {
            stopRecording();
            pttButton.style.backgroundColor = '';
            pttButton.style.color = '';
            pttButton.textContent = 'Push to Talk';
        }
    });
});