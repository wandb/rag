import { scrollToBottom } from './utils.js';
import { initializeAudioPlayer, processAudioChunk } from './player.js';
import { setupVisualization, clearVisualizations } from './visualizer.js';
import { setupTextInputHandlers, setupPTTHandlers } from './domEvents.js';
import { audioState } from './state.js';

export function setupHTMXEvents() {
    htmx.config.wsReconnectDelay = 'full-jitter';
    htmx.config.wsBinaryType = 'blob';

    htmx.on('htmx:wsOpen', async (evt) => {
        await initializeAudioPlayer(false);

        const voiceSelector = document.getElementById('voice-selector');
        if (voiceSelector) {
            const selectedVoice = voiceSelector.value;
            voiceSelector.disabled = true;

            if (selectedVoice) {
                const voiceConfig = {
                    type: 'voice_config',
                    voice: selectedVoice
                };
                evt.detail.socketWrapper.send(JSON.stringify(voiceConfig));
            }
        }
    });

    htmx.on('htmx:wsClose', async (evt) => {
        audioState.audioChunks = [];

        try {
            if (audioState.streamPlayer && audioState.streamPlayer.sourceNode) {
                audioState.streamPlayer.sourceNode.disconnect();
                audioState.streamPlayer.sourceNode = null;
            }

            if (audioState.audioElement) {
                audioState.audioElement.pause();
                audioState.audioElement.src = '';
                audioState.audioElement.load();
            }
        } catch (error) {
            console.error('Error cleaning up WebSocket resources:', error);
        }

        clearVisualizations();
        audioState.isPlaying = false;

        const voiceSelector = document.getElementById('voice-selector');
        if (voiceSelector) {
            voiceSelector.disabled = false;
        }
    });

    htmx.on('htmx:wsError', (evt) => {
        console.error('WebSocket error:', evt.detail.error);
    });

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

            if (typeof message === 'string' && message.includes('hx-swap-oob="beforeend"')) {
                setTimeout(() => {
                    scrollToBottom('event-log');
                    scrollToBottom('conversation-content');
                }, 0);
            }
        } catch (error) {
            // If parsing fails, it's not JSON data
        }
    });

    htmx.on('htmx:wsAfterMessage', async function (evt) {
        const message = evt.detail.message;

        try {
            const parsedMessage = typeof message === 'string' ? JSON.parse(message) : message;

            if (parsedMessage.type === 'response.created') {
                audioState.isFirstChunk = true;
                audioState.audioChunks = [];
                audioState.isAudioStreamComplete = false;

                if (audioState.streamPlayer) {
                    await audioState.streamPlayer.reset();
                }
                return;
            }

            if (parsedMessage.type === 'response.audio.done') {
                audioState.isAudioStreamComplete = true;
                if (audioState.audioChunks.length > 0) {
                    await processAudioChunk('');
                }
                return;
            }

            if (parsedMessage.type === 'audio') {
                await processAudioChunk(parsedMessage.data);
                return;
            }

            if (typeof message === 'string' && message.includes('hx-swap-oob')) {
                requestAnimationFrame(() => {
                    scrollToBottom('event-log');
                    scrollToBottom('conversation-content');
                });
            }
        } catch (error) {
            if (typeof message === 'string' && message.includes('hx-swap-oob')) {
                requestAnimationFrame(() => {
                    scrollToBottom('event-log');
                    scrollToBottom('conversation-content');
                });
            }
        }
    });

    htmx.on('htmx:beforeRequest', async function (evt) {
        const path = evt.detail.pathInfo.requestPath;
        if (path === '/connect') {
            try {
                const voiceSelector = document.getElementById('voice-selector');
                if (voiceSelector) {
                    localStorage.setItem('selectedVoice', voiceSelector.value);
                }

                if (!audioState.recorder) {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const tempContext = new AudioContext();
                    const actualSampleRate = tempContext.sampleRate;
                    tempContext.close();

                    audioState.recorder = new WavRecorder({
                        sampleRate: actualSampleRate,
                        outputToSpeakers: false
                    });
                    await audioState.recorder.begin();
                }
            } catch (error) {
                console.error('Recorder initialization error:', error);
                if (audioState.recorder) {
                    try {
                        await audioState.recorder.quit();
                        audioState.recorder = null;
                    } catch (cleanupError) {
                        console.error('Error during cleanup:', cleanupError);
                    }
                }
                evt.preventDefault();
            }
        } else if (path === '/disconnect') {
            const voiceSelector = document.getElementById('voice-selector');
            if (voiceSelector) {
                localStorage.setItem('selectedVoice', voiceSelector.value);
            }

            audioState.audioChunks = [];
            if (audioState.audioElement) {
                audioState.audioElement.removeAttribute('src');
                audioState.audioElement.load();
            }

            if (audioState.recorder) {
                try {
                    if (audioState.recorder.processor) {
                        await audioState.recorder.quit();
                    }
                    audioState.recorder = null;
                } catch (error) {
                    console.error('Error cleaning up recorder:', error);
                }
            }

            if (audioState.streamPlayer) {
                await audioState.streamPlayer.interrupt();
                audioState.streamPlayer = null;
            }
        }
    });

    htmx.on('htmx:load', function (evt) {
        setupVisualization();
        setupTextInputHandlers();
        setupPTTHandlers();
    });

    htmx.on('htmx:afterSwap', function (evt) {
        const voiceSelector = document.getElementById('voice-selector');
        if (voiceSelector) {
            const savedVoice = localStorage.getItem('selectedVoice');
            if (savedVoice) {
                voiceSelector.value = savedVoice;
            }
        }

        if (evt.detail.target.id === 'ws-container') {
            setupVisualization();
            setupTextInputHandlers();
            setupPTTHandlers();
        }

        if (evt.detail.target.id === 'event-log' || evt.detail.target.id === 'conversation-content') {
            scrollToBottom(evt.detail.target.id);
        }
    });

    htmx.on('htmx:beforeSwap', function (evt) {
        if (evt.detail.target.id === 'ws-container') {
            clearVisualizations();
        }
    });
} 