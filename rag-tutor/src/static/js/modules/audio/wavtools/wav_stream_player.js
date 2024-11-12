import { AudioAnalysis } from './audio_analysis.js';

/**
 * Plays audio streams received in raw PCM16 chunks from the browser
 * @class
 */
export class WavStreamPlayer {
  constructor({ sampleRate = 24000 } = {}) {
    this.sampleRate = sampleRate;
    this.context = null;
    this.analyser = null;
    this.sourceNode = null;
    this.audioElement = null;
  }

  async connect(audioElement) {
    this.audioElement = audioElement;
    this.context = new AudioContext({ sampleRate: this.sampleRate });

    // Create and configure analyzer
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.8;
    this.analyser.minDecibels = -90;
    this.analyser.maxDecibels = -10;

    // Connect audio element to analyzer
    this.sourceNode = this.context.createMediaElementSource(audioElement);
    this.sourceNode.connect(this.analyser);
    this.sourceNode.connect(this.context.destination);

    return true;
  }

  getFrequencies(analysisType = 'frequency', minDecibels = -100, maxDecibels = -30) {
    if (!this.analyser) {
      throw new Error('Not connected, please call .connect() first');
    }
    return AudioAnalysis.getFrequencies(
      this.analyser,
      this.sampleRate,
      null,
      analysisType,
      minDecibels,
      maxDecibels
    );
  }

  async interrupt() {
    if (this.audioElement) {
      const currentTime = this.audioElement.currentTime;
      const trackInfo = {
        trackId: this.audioElement.src || null,
        offset: Math.floor(currentTime * this.sampleRate),
        currentTime: currentTime
      };

      await this.reset();
      return trackInfo;
    }
    return null;
  }

  async reset() {
    if (this.audioElement) {
      this.audioElement.pause();
      this.audioElement.currentTime = 0;
      if (this.audioElement.src) {
        URL.revokeObjectURL(this.audioElement.src);
        this.audioElement.src = '';
      }
      this.audioElement.load();
    }
  }
}
globalThis.WavStreamPlayer = WavStreamPlayer;
