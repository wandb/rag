import {audioState} from '../../core/state.js';

export async function startRecording() {
    try {
        if (audioState.isProcessing || audioState.isRecording) {
            return;
        }

        audioState.isProcessing = true;

        // Only send cancel event if there are existing audio chunks
        const wsContainer = document.getElementById('ws-container');
        if (wsContainer && audioState.audioChunks.length > 0) {
            const eventDetail = {
                type: 'cancel',
                data: 'Cancel current audio stream'
            };

            const audioMessageEvent = new CustomEvent('audioMessage', { bubbles: true, detail: eventDetail });
            wsContainer.dispatchEvent(audioMessageEvent);
        }

        // Reset audio player before starting new recording
        if (audioState.streamPlayer) {
            await audioState.streamPlayer.reset();
        }
        audioState.audioChunks = [];
        audioState.isFirstChunk = true;

        // Reinitialize recorder if needed
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

        await audioState.recorder.record();
        audioState.isRecording = true;
    } catch (error) {
        console.error('Start recording error:', error);
        if (audioState.recorder) {
            try {
                await audioState.recorder.quit();
                audioState.recorder = null;
            } catch (cleanupError) {
                console.error('Error during cleanup:', cleanupError);
            }
        }
    } finally {
        audioState.isProcessing = false;
    }
}

export async function stopRecording() {
    if (!audioState.isRecording || audioState.isProcessing) {
        return;
    }

    try {
        audioState.isProcessing = true;

        if (!audioState.recorder) {
            return;
        }

        const result = await audioState.recorder.end();
        audioState.isRecording = false;

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

            audioState.recorder = new WavRecorder({
                sampleRate: actualSampleRate,
                outputToSpeakers: false
            });
            await audioState.recorder.begin();
        } catch (initError) {
            console.error('Failed to reinitialize recorder:', initError);
            audioState.recorder = null;
        }

    } catch (error) {
        console.error('Stop recording error:', error);
        if (audioState.recorder) {
            try {
                await audioState.recorder.quit();
                audioState.recorder = null;
            } catch (cleanupError) {
                console.error('Error during cleanup:', cleanupError);
            }
        }
    } finally {
        audioState.isProcessing = false;
    }
}

export function getRecorder() {
    return audioState.recorder;
}

export function isCurrentlyRecording() {
    return audioState.isRecording;
} 