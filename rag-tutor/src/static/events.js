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
        return;
    }

    try {
        isProcessing = true;

        if (!recorder) {
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


// Add state tracking at the top level
let isAudioStreamComplete = false;

async function processAudioChunk(base64Data) {
    try {
        if (!streamPlayer) {
            await initializeAudioPlayer(true);
        }

        // For the first chunk, reset everything
        if (isFirstChunk) {
            if (streamPlayer) {
                await streamPlayer.reset();
            }
            isPlaying = true;
            isFirstChunk = false;
            audioChunks = [];
            isAudioStreamComplete = false; // Reset completion flag

            // Clean up old URL
            if (audioElement.src) {
                URL.revokeObjectURL(audioElement.src);
                audioElement.src = '';
            }
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

        // Calculate buffered audio duration
        const totalSamples = audioChunks.reduce((acc, chunk) => acc + (chunk.length / 2), 0);
        const bufferedSeconds = totalSamples / OPENAI_SAMPLE_RATE;

        // Minimum buffer size before starting playback (250ms)
        const MIN_BUFFER_SIZE = 0.25;

        // Update conditions:
        // 1. Initial buffer accumulation
        // 2. Regular updates during playback
        const hasMinimumBuffer = bufferedSeconds >= MIN_BUFFER_SIZE;
        const isFirstPlay = !audioElement.duration;
        const currentPlaybackTime = audioElement.currentTime || 0;
        const currentDuration = audioElement.duration || 0;

        // Should update if:
        // 1. First play with minimum buffer
        // 2. Stream is complete (final update)
        // 3. Near end of current playback with new data
        const shouldUpdate = isFirstPlay ? hasMinimumBuffer : (
            isAudioStreamComplete || (
                currentDuration - currentPlaybackTime < 0.5 &&
                bufferedSeconds > currentDuration + 0.2
            )
        );

        if (shouldUpdate) {
            const wavHeader = createWavHeader(audioChunks);
            const completeChunks = [wavHeader, ...audioChunks];
            const audioBlob = new Blob(completeChunks, { type: 'audio/wav' });

            const currentTime = audioElement.currentTime;
            const wasPlaying = !audioElement.paused && !audioElement.ended;

            // Create new URL and clean up old one
            const url = URL.createObjectURL(audioBlob);
            const oldSrc = audioElement.src;

            // Brief pause to avoid interruption
            if (wasPlaying) {
                await audioElement.pause();
            }

            // Update source
            audioElement.src = url;
            if (oldSrc) {
                setTimeout(() => URL.revokeObjectURL(oldSrc), 1000);
            }

            // Restore playback state
            if (wasPlaying || isFirstPlay) {
                try {
                    // Ensure audio is loaded before playing
                    await new Promise((resolve) => {
                        audioElement.addEventListener('loadedmetadata', resolve, { once: true });
                    });

                    await audioElement.play();

                    // Restore position if we were already playing
                    if (currentTime > 0 && !isFirstPlay) {
                        audioElement.currentTime = currentTime;
                    }
                } catch (playError) {
                    console.warn('Playback restoration failed:', playError);
                }
            }
        }

    } catch (error) {
        console.error('Error processing audio chunk:', error);
    }
}

// Helper function to create WAV header
function createWavHeader(dataLength) {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);

    // If dataLength is an array of chunks, sum their lengths
    const totalLength = Array.isArray(dataLength)
        ? dataLength.reduce((acc, chunk) => acc + chunk.length, 0)
        : dataLength;

    // "RIFF" chunk descriptor
    view.setUint32(0, 0x52494646, false); // "RIFF"
    view.setUint32(4, 36 + totalLength, true); // File size
    view.setUint32(8, 0x57415645, false); // "WAVE"

    // "fmt " sub-chunk
    view.setUint32(12, 0x666D7420, false); // "fmt "
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true); // NumChannels (1 for mono)
    view.setUint32(24, OPENAI_SAMPLE_RATE, true); // SampleRate
    view.setUint32(28, OPENAI_SAMPLE_RATE * 2, true); // ByteRate
    view.setUint16(32, 2, true); // BlockAlign
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
    if (triggerEvent) {
        if (triggerEvent.type === 'audioMessage') {
            evt.detail.messageBody = JSON.stringify({
                type: 'audio',
                ...triggerEvent.detail
            });
        } else if (triggerEvent.type === 'textMessage') {
            evt.detail.messageBody = JSON.stringify({
                type: 'text',
                ...triggerEvent.detail
            });
        }
    }
});


htmx.on('htmx:wsBeforeMessage', async function (evt) {
    const message = evt.detail.message;

    try {
        const parsedMessage = typeof message === 'string' ? JSON.parse(message) : message;

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

htmx.on('htmx:wsAfterMessage', async function (evt) {
    const message = evt.detail.message;

    try {
        const parsedMessage = typeof message === 'string' ? JSON.parse(message) : message;

        if (parsedMessage.type === 'response.created') {
            // Reset audio state for new response
            isFirstChunk = true;
            audioChunks = [];
            isAudioStreamComplete = false;  // Reset completion flag

            // Reset the stream player and audio element
            if (streamPlayer) {
                await streamPlayer.reset();
            }
            return;
        }

        // Handle audio completion event
        if (parsedMessage.type === 'response.audio.done') {
            isAudioStreamComplete = true;
            // Force final update of audio player
            if (audioChunks.length > 0) {
                await processAudioChunk(''); // Empty chunk to trigger final update
            }
            return;
        }

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

// Add these functions at the top level
function handleTextSend() {
    const sendButton = document.getElementById('send-btn');
    const textInput = document.getElementById('text-input');

    if (sendButton && textInput && !sendButton.disabled && textInput.value.trim()) {
        const wsContainer = document.getElementById('ws-container');

        if (wsContainer) {
            const eventDetail = {
                type: 'text',
                data: textInput.value.trim()
            };

            const textMessageEvent = new CustomEvent('textMessage', {
                bubbles: true,
                detail: eventDetail
            });
            wsContainer.dispatchEvent(textMessageEvent);

            // Clear the input after sending
            textInput.value = '';
        }
    }
}

function setupTextInputHandlers() {
    const sendButton = document.getElementById('send-btn');
    const textInput = document.getElementById('text-input');

    // Remove existing listeners first to prevent duplicates
    if (sendButton) {
        sendButton.removeEventListener('click', handleTextSend);
        sendButton.addEventListener('click', handleTextSend);
    }

    if (textInput) {
        const keydownHandler = function (event) {
            if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                handleTextSend();
            }
        };
        textInput.removeEventListener('keydown', keydownHandler);
        textInput.addEventListener('keydown', keydownHandler);
    }
}

// Update the existing htmx:afterSwap handler
htmx.on('htmx:afterSwap', function (evt) {

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

    // Set up text input handlers after swap
    setupTextInputHandlers();
});

// Update the existing DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function () {
    // Set up text input handlers
    setupTextInputHandlers();

    // Existing PTT button event listeners
    function handlePTTStart(pttButton) {
        if (pttButton && !pttButton.disabled && !isProcessing) {
            startRecording();
            pttButton.style.backgroundColor = '#ff4444';
            pttButton.style.color = 'white';
            pttButton.textContent = 'Release to Send';
        }
    }

    function handlePTTEnd(pttButton) {
        if (pttButton && !pttButton.disabled && isRecording) {
            stopRecording();
            pttButton.style.backgroundColor = '';
            pttButton.style.color = '';
            pttButton.textContent = 'Push to Talk';
        }
    }

    // Mouse events
    document.addEventListener('mousedown', function (event) {
        const pttButton = event.target.closest('#ptt-btn');
        handlePTTStart(pttButton);
    });

    document.addEventListener('mouseup', function (event) {
        const pttButton = event.target.closest('#ptt-btn');
        handlePTTEnd(pttButton);
    });

    // Keyboard events
    let spacebarPressed = false;
    document.addEventListener('keydown', function (event) {
        // Check if the target is the text input
        const isTextInput = event.target.id === 'text-input';

        // Only handle PTT if not in text input and spacebar is pressed
        if (event.code === 'Space' && !spacebarPressed && !isTextInput) {
            // Prevent spacebar from scrolling the page
            event.preventDefault();
            spacebarPressed = true;
            const pttButton = document.getElementById('ptt-btn');
            handlePTTStart(pttButton);
        }
    });

    document.addEventListener('keyup', function (event) {
        // Only handle PTT release if not in text input
        if (event.code === 'Space' && spacebarPressed && event.target.id !== 'text-input') {
            event.preventDefault();
            spacebarPressed = false;
            const pttButton = document.getElementById('ptt-btn');
            handlePTTEnd(pttButton);
        }
    });

});

