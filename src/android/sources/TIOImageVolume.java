package com.tensorflow.fidelidade.plugin.sources;

public class TIOImageVolume {
    public int height;
    public int width;
    public int channels;

    public TIOImageVolume(int height, int width, int channels) {
        this.height = height;
        this.width = width;
        this.channels = channels;
    }
}

