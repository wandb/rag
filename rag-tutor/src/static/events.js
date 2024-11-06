// Audio recording functionality
let recorder;

// Add state tracking
let isRecording = false;
let isProcessing = false;

// Add visualization intervals tracking
const visualizationIntervals = new Map();

// Add this helper function at the top level
function scrollToBottom(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        // Use requestAnimationFrame to ensure DOM updates are complete
        requestAnimationFrame(() => {
            element.scrollTop = element.scrollHeight;
        });
    }
}

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

// Add at the top with other globals
let clientCanvas, serverCanvas;
let clientCtx, serverCtx;

function drawSineWave(canvas, ctx, values, color) {
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas with a light background
    ctx.fillStyle = '#f8f8f8';
    ctx.fillRect(0, 0, width, height);

    // Draw the center line
    ctx.beginPath();
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw the wave
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    // Normalize and animate the values
    const now = Date.now() / 1000; // Get current time in seconds
    const points = new Float32Array(values.length);

    for (let i = 0; i < values.length; i++) {
        // Add some oscillation to make it more dynamic
        const oscillation = Math.sin(now * 4 + i * 0.1) * 0.15;
        points[i] = values[i] + oscillation;
    }

    // Start from the left edge
    ctx.moveTo(0, height / 2);

    // Draw the wave with smooth curves
    for (let i = 0; i < width; i += 2) {
        const valueIndex = Math.floor((i / width) * points.length);
        const value = points[valueIndex] || 0;
        const nextValue = points[Math.min(valueIndex + 1, points.length - 1)] || 0;

        // Calculate current and next points with animation
        const normalizedValue = (value - 0.5) * 0.8; // Reduced amplitude
        const normalizedNextValue = (nextValue - 0.5) * 0.8;

        const x = i;
        const nextX = Math.min(i + 2, width);
        const y = (height / 2) + (normalizedValue * height / 2);
        const nextY = (height / 2) + (normalizedNextValue * height / 2);

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            // Use quadratic curve with control point halfway between points
            const controlX = (x + nextX) / 2;
            const controlY = (y + nextY) / 2;
            ctx.quadraticCurveTo(x, y, controlX, controlY);
        }
    }

    ctx.stroke();

}

function setupVisualization() {
    clientCanvas = document.getElementById('client-canvas');
    serverCanvas = document.getElementById('server-canvas');

    if (clientCanvas && serverCanvas) {
        clientCtx = clientCanvas.getContext('2d');
        serverCtx = serverCanvas.getContext('2d');

        // Set initial canvas dimensions
        function resizeCanvas(canvas) {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }

        resizeCanvas(clientCanvas);
        resizeCanvas(serverCanvas);

        // Add window resize handler
        window.addEventListener('resize', () => {
            resizeCanvas(clientCanvas);
            resizeCanvas(serverCanvas);
        });

        function render() {
            if (!clientCanvas || !serverCanvas) return;

            // Client-side visualization (recording)
            if (recorder && recorder.processor) {
                try {
                    const result = recorder.getFrequencies('voice');
                    drawSineWave(clientCanvas, clientCtx, isRecording ? result.values : new Float32Array(128).fill(0.5), '#0099ff');
                } catch (error) {
                    drawSineWave(clientCanvas, clientCtx, new Float32Array(128).fill(0.5), '#0099ff');
                }
            } else {
                drawSineWave(clientCanvas, clientCtx, new Float32Array(128).fill(0.5), '#0099ff');
            }

            // Server-side visualization (playback)
            if (streamPlayer && streamPlayer.analyser && isPlaying) {
                try {
                    const result = streamPlayer.getFrequencies('voice');
                    // Add some randomness to make it more dynamic when there's no actual audio
                    const values = result.values.map(v =>
                        v === 0.5 ? 0.5 + (Math.random() * 0.01 - 0.005) : v
                    );
                    drawSineWave(serverCanvas, serverCtx, values, '#009900');
                } catch (error) {
                    console.error('Visualization error:', error);
                    drawSineWave(serverCanvas, serverCtx, new Float32Array(128).fill(0.5), '#009900');
                }
            } else {
                drawSineWave(serverCanvas, serverCtx, new Float32Array(128).fill(0.5), '#009900');
            }

            requestAnimationFrame(render);
        }

        render();
    }
}
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

        // Add visualization setup
        setupVisualization();

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

                // Only create the media element source connection if it hasn't been done before
                if (audioElement && streamPlayer.context && !streamPlayer.sourceNode) {
                    streamPlayer.sourceNode = streamPlayer.context.createMediaElementSource(audioElement);
                    streamPlayer.sourceNode.connect(streamPlayer.analyser);
                    streamPlayer.sourceNode.connect(streamPlayer.context.destination);
                }
            }
            isPlaying = true;
            isFirstChunk = false;
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
        if (uint8Array.length < 4096) {  // Assuming small chunks are final chunks
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
// htmx.on('htmx:wsConnecting', (evt) => {
//     // Get the voice selection
//     const voiceSelector = document.getElementById('voice-selector');
//     if (voiceSelector) {
//         const selectedVoice = voiceSelector.value;
//         // Store the selected voice for use after connection
//         evt.detail.elt.dataset.selectedVoice = selectedVoice;
//     }
// });

htmx.on('htmx:wsOpen', async (evt) => {
    await initializeAudioPlayer(false);

    // Send the voice configuration message
    const voiceSelector = document.getElementById('voice-selector');
    if (voiceSelector) {
        const selectedVoice = voiceSelector.value;
        // Disable the selector after connection
        voiceSelector.disabled = true;

        if (selectedVoice) {
            const voiceConfig = {
                type: 'voice_config',
                voice: selectedVoice
            };
            // Send the configuration via WebSocket
            evt.detail.socketWrapper.send(JSON.stringify(voiceConfig));
        }
    }
});

// Clean up when WebSocket closes
htmx.on('htmx:wsClose', async (evt) => {
    // console.log('WebSocket connection closed');
    audioChunks = []; // Clear stored chunks

    try {
        if (streamPlayer && streamPlayer.sourceNode) {
            streamPlayer.sourceNode.disconnect();
            streamPlayer.sourceNode = null;
        }

        if (audioElement) {
            audioElement.pause();
            audioElement.src = '';
            audioElement.load();
        }
    } catch (error) {
        console.error('Error cleaning up WebSocket resources:', error);
    }

    // Clear visualization
    if (clientCtx && clientCanvas) {
        clientCtx.clearRect(0, 0, clientCanvas.width, clientCanvas.height);
    }
    if (serverCtx && serverCanvas) {
        serverCtx.clearRect(0, 0, serverCanvas.width, serverCanvas.height);
    }

    isPlaying = false;  // Reset playing state

    // Re-enable the voice selector when connection closes
    const voiceSelector = document.getElementById('voice-selector');
    if (voiceSelector) {
        voiceSelector.disabled = false;
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

        // If it's an HTML update with hx-swap-oob="beforeend"
        if (typeof message === 'string' && message.includes('hx-swap-oob="beforeend"')) {
            // Schedule scroll after the DOM update
            setTimeout(() => {
                scrollToBottom('event-log');
                scrollToBottom('conversation-content');
            }, 0);
        }
    } catch (error) {
        // If parsing fails, it's not JSON data, let HTMX handle it silently
    }

    // Let HTMX handle non-audio messages for DOM updates
});

// Update the htmx:wsAfterMessage handler
htmx.on('htmx:wsAfterMessage', async function (evt) {
    const message = evt.detail.message;

    try {
        // Try to parse the message if it's a string
        const parsedMessage = typeof message === 'string' ? JSON.parse(message) : message;

        // Handle audio messages
        if (parsedMessage.type === 'audio') {
            await processAudioChunk(parsedMessage.data);
            return;
        }

        // If it's an HTML update
        if (typeof message === 'string' && message.includes('hx-swap-oob')) {
            // Use requestAnimationFrame to ensure DOM is updated
            requestAnimationFrame(() => {
                scrollToBottom('event-log');
                scrollToBottom('conversation-content');
            });
        }
    } catch (error) {
        // If parsing fails, it's not JSON data
        if (typeof message === 'string' && message.includes('hx-swap-oob')) {
            requestAnimationFrame(() => {
                scrollToBottom('event-log');
                scrollToBottom('conversation-content');
            });
        }
    }
});

// Other HTMX Events
htmx.on('htmx:beforeRequest', async function (evt) {
    const path = evt.detail.pathInfo.requestPath;
    if (path === '/connect') {
        try {
            // Store the current voice selection before the request
            const voiceSelector = document.getElementById('voice-selector');
            if (voiceSelector) {
                localStorage.setItem('selectedVoice', voiceSelector.value);
            }

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
        // Store the current voice selection before disconnecting
        const voiceSelector = document.getElementById('voice-selector');
        if (voiceSelector) {
            localStorage.setItem('selectedVoice', voiceSelector.value);
        }

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

    // Restore voice selection if available
    const voiceSelector = document.getElementById('voice-selector');
    if (voiceSelector) {
        const savedVoice = localStorage.getItem('selectedVoice');
        if (savedVoice) {
            voiceSelector.value = savedVoice;
        }
    }

    // Check if the swapped element is either the event log or conversation content
    if (evt.detail.target.id === 'event-log' || evt.detail.target.id === 'conversation-content') {
        scrollToBottom(evt.detail.target.id);
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

