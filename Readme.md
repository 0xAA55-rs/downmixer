# Downmixer

The downmixer to downmix multi-channels audio to stereo

## Overview

The downmixer needs a `DownmixerParams` to specify the modifier of each channels by dB values.

If you have no idea about the dB modifying, use `DownmixerParams::default()`.

The default value is:
```rust
impl DownmixerParams {
    /// * Setup default parameters
    pub fn new() -> Self {
        Self {
            front_lr_db: 0.0,
            front_center_db: -3.0,
            lowfreq_db: -6.0,
            back_lr_db: -3.0,
            front_center_lr_db: -1.5,
            back_center_db: -4.5,
            side_lr_db: -3.0,
            top_center_db: -4.5,
            top_front_lr_db: -3.0,
            top_front_center_db: -4.5,
            top_back_lr_db: -3.0,
            top_back_center_db: -4.5,
        }
    }
}
```

### The implementation of `Downmixer`

```rust
#[derive(Debug, Clone, Copy)]
pub struct Downmixer {
    /// Num channels
    pub channels: u16,

    /// The channel mask indicates which channel is for which speaker.
    pub channel_mask: u32,

    /// The weights for downmixing, the `u32` is the bitmask indicating which speaker the channel is, and the `f64` is the weight.
    pub gains: CopiableBuffer<(u32, f64), 18>,
}

impl Downmixer {
    /// * Create a new `Downmixer` by specifying the channel mask and the `DownmixerParams` to compute gains for each channel.
    pub fn new(channel_mask: u32, params: DownmixerParams) -> Self {
        let mut ret = Self {
            channels: 0,
            channel_mask,
            gains: CopiableBuffer::new(),
        };
        ret.gains = params.gains_from_channel_mask(channel_mask).into_iter().collect();
        ret.channels = ret.gains.len() as u16;
        ret
    }

    /// * Downmix an audio frame to a stereo frame.
    pub fn downmix_frame_to_stereo<S>(&self, frame: &[S]) -> (S, S)
    where
        S: SampleType {
        use speaker_positions::*;
        let lmax: f64 = self.gains.iter().map(|&(b, g)| if is_lcenter(b) {g} else {0.0}).sum();
        let rmax: f64 = self.gains.iter().map(|&(b, g)| if is_rcenter(b) {g} else {0.0}).sum();
        let lmix: f64 = self.gains.iter().enumerate().map(|(i, &(b, g))| if is_lcenter(b) {frame[i].to_f64() * g} else {0.0}).sum();
        let rmix: f64 = self.gains.iter().enumerate().map(|(i, &(b, g))| if is_rcenter(b) {frame[i].to_f64() * g} else {0.0}).sum();
        (S::scale_from(lmix / lmax), S::scale_from(rmix / rmax))
    }

    /// Downmix multiple audio frames to stereo frames
    pub fn downmix_frame_to_stereos<S>(&self, frames: &[Vec<S>]) -> Vec<(S, S)>
    where
        S: SampleType {
        frames.iter().map(|frame|self.downmix_frame_to_stereo(frame)).collect()
    }

    /// * Downmix an audio frame to a mono frame.
    pub fn downmix_frame_to_mono<S>(&self, frame: &[S]) -> S
    where
        S: SampleType {
        let (l, r) = self.downmix_frame_to_stereo(frame);
        S::average(l, r)
    }
}
```
