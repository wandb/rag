// Audio recording functionality
let recorder;

// Add WavStreamPlayer initialization and handling
let streamPlayer = null;
let visualizationInterval = null;

async function initializeStreamPlayer() {
    if (!streamPlayer) {
        streamPlayer = new WavStreamPlayer({ sampleRate: 44100 });
        await streamPlayer.connect();
    }
}

function updateVisualization(canvas) {
    const ctx = canvas.getContext('2d');
    const frequencies = streamPlayer.getFrequencies('frequency');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw visualization
    ctx.beginPath();
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 2;

    const barWidth = canvas.width / frequencies.length;
    frequencies.forEach((freq, i) => {
        const height = (freq + 100) * canvas.height / 70; // Normalize values
        const x = i * barWidth;
        ctx.moveTo(x, canvas.height);
        ctx.lineTo(x, canvas.height - height);
    });
    ctx.stroke();
}

window.playAudioChunk = async function (container) {
    await initializeStreamPlayer();

    const base64Data = container.getAttribute('data-audio');
    const canvas = container.querySelector('.audio-visualizer');
    const playBtn = container.querySelector('.play-btn');

    // Convert and play audio
    const binaryString = window.atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    const pcmData = new Int16Array(bytes.buffer);

    // Start visualization
    if (visualizationInterval) {
        clearInterval(visualizationInterval);
    }
    visualizationInterval = setInterval(() => updateVisualization(canvas), 50);

    // Update button state
    playBtn.textContent = "🔊 Playing...";
    playBtn.disabled = true;

    // Play audio
    streamPlayer.add16BitPCM(pcmData);

    // Reset button after approximate playback duration
    setTimeout(() => {
        playBtn.textContent = "▶️ Play";
        playBtn.disabled = false;
        clearInterval(visualizationInterval);
    }, (pcmData.length / 44100) * 1000); // Duration based on sample rate
};

// Configure WebSocket behavior
htmx.config.wsReconnectDelay = 'full-jitter';
htmx.config.wsBinaryType = 'blob';

// Move all HTMX event listeners outside DOMContentLoaded
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
        const wsContainer = document.getElementById('ws-container');
        if (wsContainer) {
            wsContainer.removeAttribute('hx-ext');
            wsContainer.removeAttribute('ws-connect');
        }

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
    }
});

htmx.on('htmx:afterSwap', function (evt) {
    const pttButton = document.getElementById('ptt-btn');
    if (pttButton) {
        console.log('Button swapped, new disabled state:', pttButton.disabled);
    }
    if (evt.detail.target.id === 'event-log') {
        evt.detail.target.scrollTop = evt.detail.target.scrollHeight;
    }
});

// WebSocket event handlers
htmx.on('htmx:wsConnecting', (evt) => {
    console.log('Connecting to WebSocket...', evt.detail.elt.id);
});

htmx.on('htmx:wsOpen', (evt) => {
    console.log('WebSocket Connected', evt.detail.elt.id);
});

htmx.on('htmx:wsConfigSend', (evt) => {
    console.log('htmx:wsConfigSend event received:', evt);
    const triggerEvent = evt.detail.triggeringEvent;
    if (triggerEvent && triggerEvent.type === 'audioMessage') {
        evt.detail.messageBody = JSON.stringify({
            type: 'audio',
            ...triggerEvent.detail
        });
        console.log('Configured WebSocket message:', evt.detail.messageBody.substring(0, 100) + '...');
    }
});

htmx.on('htmx:wsBeforeSend', (evt) => {
    console.log('Sending audio data from elt:', evt.detail.elt);
});

htmx.on('htmx:wsAfterSend', (evt) => {
    console.log('Audio data sent successfully from elt:', evt.detail.elt);
});

htmx.on('htmx:wsClose', (evt) => {
    console.error('WebSocket connection closed', evt.detail);
});

htmx.on('htmx:wsError', (evt) => {
    console.error('WebSocket error:', evt.detail.error);
    // console.error('Error event:', evt);
});

// Recording functions
async function startRecording() {
    try {
        if (!recorder) {
            console.error('Recorder not initialized');
            return;
        }
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
        const result = await recorder.end();
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
        } else {
            console.error('WebSocket container not found');
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
    }
}

// DOM event listeners
document.addEventListener('DOMContentLoaded', function () {
    // PTT button event listeners
    document.addEventListener('mousedown', function (event) {
        const pttButton = event.target.closest('#ptt-btn');
        if (pttButton && !pttButton.disabled) {
            startRecording();
            pttButton.style.backgroundColor = '#ff4444';
            pttButton.style.color = 'white';
            pttButton.textContent = 'Release to Send';
        }
    });

    document.addEventListener('mouseup', function (event) {
        const pttButton = event.target.closest('#ptt-btn');
        if (pttButton && !pttButton.disabled) {
            stopRecording();
            pttButton.style.backgroundColor = '';
            pttButton.style.color = '';
            pttButton.textContent = 'Push to Talk';
        }
    });
});