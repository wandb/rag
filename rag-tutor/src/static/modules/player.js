import { createWavHeader } from './utils.js';
import { audioState } from './state.js';

const OPENAI_SAMPLE_RATE = 24000;  // OpenAI uses 24kHz

export async function initializeAudioPlayer(autoplay = true) {
    try {
        audioState.audioElement = document.getElementById('audio-player');
        if (!audioState.audioElement) {
            console.error('Audio element not found');
            return;
        }

        // Initialize WavStreamPlayer with correct sample rate
        audioState.streamPlayer = new WavStreamPlayer({
            sampleRate: OPENAI_SAMPLE_RATE
        });
        await audioState.streamPlayer.connect(audioState.audioElement);

        // Set audio element properties
        audioState.audioElement.autoplay = true;
        audioState.audioElement.muted = false;

        // Try to enable autoplay
        if (autoplay) {
            try {
                await audioState.audioElement.play();
            } catch (e) {
                const interactionEvents = ['click', 'touchstart', 'keydown'];
                const playHandler = async () => {
                    try {
                        await audioState.streamPlayer.context.resume();
                        await audioState.audioElement.play();
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

export async function processAudioChunk(base64Data) {
    try {
        if (!audioState.streamPlayer) {
            await initializeAudioPlayer(true);
        }

        // For the first chunk, reset everything
        if (audioState.isFirstChunk) {
            if (audioState.streamPlayer) {
                await audioState.streamPlayer.reset();
            }
            audioState.isPlaying = true;
            audioState.isFirstChunk = false;
            audioState.audioChunks = [];
            audioState.isAudioStreamComplete = false;

            // Clean up old URL
            if (audioState.audioElement.src) {
                URL.revokeObjectURL(audioState.audioElement.src);
                audioState.audioElement.src = '';
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
        audioState.audioChunks.push(uint8Array);

        // Calculate buffered audio duration
        const totalSamples = audioState.audioChunks.reduce((acc, chunk) => acc + (chunk.length / 2), 0);
        const bufferedSeconds = totalSamples / OPENAI_SAMPLE_RATE;

        // Minimum buffer size before starting playback (250ms)
        const MIN_BUFFER_SIZE = 0.25;

        const hasMinimumBuffer = bufferedSeconds >= MIN_BUFFER_SIZE;
        const isFirstPlay = !audioState.audioElement.duration;
        const currentPlaybackTime = audioState.audioElement.currentTime || 0;
        const currentDuration = audioState.audioElement.duration || 0;

        const shouldUpdate = isFirstPlay ? hasMinimumBuffer : (
            audioState.isAudioStreamComplete || (
                currentDuration - currentPlaybackTime < 0.5 &&
                bufferedSeconds > currentDuration + 0.2
            )
        );

        if (shouldUpdate) {
            const wavHeader = createWavHeader(audioState.audioChunks);
            const completeChunks = [wavHeader, ...audioState.audioChunks];
            const audioBlob = new Blob(completeChunks, { type: 'audio/wav' });

            const currentTime = audioState.audioElement.currentTime;
            const wasPlaying = !audioState.audioElement.paused && !audioState.audioElement.ended;

            const url = URL.createObjectURL(audioBlob);
            const oldSrc = audioState.audioElement.src;

            if (wasPlaying) {
                await audioState.audioElement.pause();
            }

            audioState.audioElement.src = url;
            if (oldSrc) {
                setTimeout(() => URL.revokeObjectURL(oldSrc), 1000);
            }

            if (wasPlaying || isFirstPlay) {
                try {
                    await new Promise((resolve) => {
                        audioState.audioElement.addEventListener('loadedmetadata', resolve, { once: true });
                    });

                    await audioState.audioElement.play();

                    if (currentTime > 0 && !isFirstPlay) {
                        audioState.audioElement.currentTime = currentTime;
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