import { startRecording, stopRecording } from '../modules/audio/recorder.js';
import { audioState } from './state.js';

export async function handleTextSend() {
    const sendButton = document.getElementById('send-btn');
    const textInput = document.getElementById('text-input');

    if (sendButton && textInput && !sendButton.disabled && textInput.value.trim()) {
        const wsContainer = document.getElementById('ws-container');

        if (wsContainer) {
            // First send cancel event if audio is playing
            if (audioState.streamPlayer && audioState.audioElement && !audioState.audioElement.paused) {
                const cancelEventDetail = {
                    type: 'cancel',
                    data: 'Cancel current audio stream'
                };
                const cancelEvent = new CustomEvent('audioMessage', {
                    bubbles: true,
                    detail: cancelEventDetail
                });
                wsContainer.dispatchEvent(cancelEvent);

                // Wait for audio to stop
                await audioState.streamPlayer.reset();
                // Ensure the audio element is paused
                audioState.audioElement.pause();
                audioState.audioElement.currentTime = 0;
            }

            // Then send the text message
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

export function setupTextInputHandlers() {
    const sendButton = document.getElementById('send-btn');
    const textInput = document.getElementById('text-input');

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

export function setupPTTHandlers() {
    async function handlePTTStart(pttButton) {
        if (pttButton && !pttButton.disabled && !audioState.isProcessing) {
            // Pause and reset audio playback immediately
            if (audioState.streamPlayer) {
                await audioState.streamPlayer.reset();
            }

            if (audioState.audioElement) {
                audioState.audioElement.pause();
                audioState.audioElement.currentTime = 0; // Reset to start
                if (audioState.audioElement.src) {
                    URL.revokeObjectURL(audioState.audioElement.src);
                    audioState.audioElement.src = '';
                }
                audioState.audioElement.load();
            }

            startRecording();
            pttButton.style.backgroundColor = '#ff4444';
            pttButton.style.color = 'white';
            pttButton.textContent = 'Release to Send';
        }
    }

    function handlePTTEnd(pttButton) {
        if (pttButton && !pttButton.disabled && audioState.isRecording) {
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
        const isTextInput = event.target.id === 'text-input';
        if (event.code === 'Space' && !spacebarPressed && !isTextInput) {
            event.preventDefault();
            spacebarPressed = true;
            const pttButton = document.getElementById('ptt-btn');
            handlePTTStart(pttButton);
        }
    });

    document.addEventListener('keyup', function (event) {
        if (event.code === 'Space' && spacebarPressed && event.target.id !== 'text-input') {
            event.preventDefault();
            spacebarPressed = false;
            const pttButton = document.getElementById('ptt-btn');
            handlePTTEnd(pttButton);
        }
    });
} 