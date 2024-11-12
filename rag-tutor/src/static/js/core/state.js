// Shared audio state
export const audioState = {
    audioChunks: [],
    isFirstChunk: true,
    isAudioStreamComplete: false,
    isPlaying: false,
    streamPlayer: null,
    audioElement: null,
    recorder: null,
    isRecording: false,
    isProcessing: false
}; 