#![allow(dead_code)]
use crate::AudioError;
/// Algorithm Design:
/// 1. Spatial Mapping:
///    - Assign a 3D direction vector to each audio source, representing its position relative to the listener's head.
///    - Vectors are normalized (magnitude = 1.0) to abstract distance, focusing on angular positioning.
///
/// 2. Directional Influence Calculation:
///    - Compute dot products between each source vector and the listener's facing direction (head orientation vector).
///    - Sources behind the listener (dot product < 0.0) are attenuated by a decay factor (e.g., 0.2x gain).
///
/// 3. Energy-Preserving Mixdown:
///    - Apply weighted summation: mixed_sample = Σ (source_sample * dot_product * decay_factor)
///    - Normalize weights dynamically to ensure Σ (effective_gain) ≤ 1.0, preventing clipping.
///
/// This achieves lossless channel layout conversion (e.g., 5.1 → stereo) with spatial accuracy.
///
/// Documents: <https://professionalsupport.dolby.com/s/article/How-do-the-5-1-and-Stereo-downmix-settings-work?language=en_US>
use crate::SampleType;
use crate::copiablebuf::CopiableBuffer;

/// * Convert dB modification to gain
#[inline(always)]
pub fn db_to_gain(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// * Convert gain to dB modification
#[inline(always)]
pub fn gain_to_db(gain: f64) -> f64 {
    gain.log10() * 20.0
}

/// * Modify the dB of the sample.
pub fn modify_db<S>(samples: &[S], db: f64) -> Vec<S>
where
    S: SampleType,
{
    modify_gain(samples, db_to_gain(db))
}

/// * Modify the dB of the sample.
pub fn modify_gain<S>(samples: &[S], gain: f64) -> Vec<S>
where
    S: SampleType,
{
    samples
        .iter()
        .map(|s| S::cast_from(s.as_f64() * gain))
        .collect()
}

/// * Speaker position bit mask data for multi-channel audio.
/// * This also be used on single-channel audio or double-channel audio.
#[allow(non_upper_case_globals)]
pub mod speaker_positions {
    use crate::AudioError;

    pub const FRONT_LEFT: u32 = 0x1;
    pub const FRONT_RIGHT: u32 = 0x2;
    pub const FRONT_CENTER: u32 = 0x4;
    pub const LOW_FREQ: u32 = 0x8;
    pub const BACK_LEFT: u32 = 0x10;
    pub const BACK_RIGHT: u32 = 0x20;
    pub const FRONT_LEFT_OF_CENTER: u32 = 0x40;
    pub const FRONT_RIGHT_OF_CENTER: u32 = 0x80;
    pub const BACK_CENTER: u32 = 0x100;
    pub const SIDE_LEFT: u32 = 0x200;
    pub const SIDE_RIGHT: u32 = 0x400;
    pub const TOP_CENTER: u32 = 0x800;
    pub const TOP_FRONT_LEFT: u32 = 0x1000;
    pub const TOP_FRONT_CENTER: u32 = 0x2000;
    pub const TOP_FRONT_RIGHT: u32 = 0x4000;
    pub const TOP_BACK_LEFT: u32 = 0x8000;
    pub const TOP_BACK_CENTER: u32 = 0x10000;
    pub const TOP_BACK_RIGHT: u32 = 0x20000;

    /// * The channel mask for mono audio layout
    pub const MONO_LAYOUT: u32 = FRONT_CENTER;

    /// * The channel mask for stereo audio layout
    pub const STEREO_LAYOUT: u32 = FRONT_LEFT | FRONT_RIGHT;

    /// * The channel mask for dolby 5.1 audio layout
    pub const DOLBY_5_1_FRONT_BACK_LAYOUT: u32 = FRONT_LEFT
        | FRONT_RIGHT
        | FRONT_CENTER
        | BACK_LEFT
        | BACK_RIGHT
        | LOW_FREQ;
    pub const DOLBY_5_1_FRONT_SIDE_LAYOUT: u32 = FRONT_LEFT
        | FRONT_RIGHT
        | FRONT_CENTER
        | SIDE_LEFT
        | SIDE_RIGHT
        | LOW_FREQ;

    /// * The channel mask for dolby 7.1 audio layout
    pub const DOLBY_7_1_LAYOUT: u32 = FRONT_LEFT
        | FRONT_RIGHT
        | FRONT_CENTER
        | SIDE_LEFT
        | SIDE_RIGHT
        | BACK_LEFT
        | BACK_RIGHT
        | LOW_FREQ;

    /// * The channel masks only for center channels
    pub const CENTER_BITS: u32 = FRONT_CENTER
        | BACK_CENTER
        | LOW_FREQ
        | TOP_CENTER
        | TOP_FRONT_CENTER
        | TOP_BACK_CENTER;

    /// * The channel masks only for left channels
    pub const LEFT_BITS: u32 = FRONT_LEFT
        | BACK_LEFT
        | FRONT_LEFT_OF_CENTER
        | SIDE_LEFT
        | TOP_FRONT_LEFT
        | TOP_BACK_LEFT;

    /// * The channel masks only for right channels
    pub const RIGHT_BITS: u32 = FRONT_RIGHT
        | BACK_RIGHT
        | FRONT_RIGHT_OF_CENTER
        | SIDE_RIGHT
        | TOP_FRONT_RIGHT
        | TOP_BACK_RIGHT;

    /// * The channel masks only for side channels
    pub const SIDE_BITS: u32 = LEFT_BITS | RIGHT_BITS;

    pub fn is_center(channel_bit: u32) -> bool {
        (channel_bit & CENTER_BITS) != 0
    }

    pub fn is_side(channel_bit: u32) -> bool {
        (channel_bit & SIDE_BITS) != 0
    }

    pub fn is_left(channel_bit: u32) -> bool {
        (channel_bit & LEFT_BITS) != 0
    }

    pub fn is_right(channel_bit: u32) -> bool {
        (channel_bit & RIGHT_BITS) != 0
    }

    pub fn is_lcenter(channel_bit: u32) -> bool {
        (channel_bit & (LEFT_BITS | CENTER_BITS)) != 0
    }

    pub fn is_rcenter(channel_bit: u32) -> bool {
        (channel_bit & (RIGHT_BITS | CENTER_BITS)) != 0
    }

    pub fn channel_bit_to_string(bit: u32) -> &'static str {
        match bit {
            FRONT_LEFT => "front_left",
            FRONT_RIGHT => "front_right",
            FRONT_CENTER => "front_center",
            LOW_FREQ => "low_freq",
            BACK_LEFT => "back_left",
            BACK_RIGHT => "back_right",
            FRONT_LEFT_OF_CENTER => "front_left_of_center",
            FRONT_RIGHT_OF_CENTER => "front_right_of_center",
            BACK_CENTER => "back_center",
            SIDE_LEFT => "side_left",
            SIDE_RIGHT => "side_right",
            TOP_CENTER => "top_center",
            TOP_FRONT_LEFT => "top_front_left",
            TOP_FRONT_CENTER => "top_front_center",
            TOP_FRONT_RIGHT => "top_front_right",
            TOP_BACK_LEFT => "top_back_left",
            TOP_BACK_CENTER => "top_back_center",
            TOP_BACK_RIGHT => "top_back_right",
            _ => "Invalid bit",
        }
    }

    pub fn channel_mask_to_string(channel_mask: u32) -> String {
        channel_mask_to_speaker_positions_descs(channel_mask).join(" + ")
    }

    /// * Break down `channel_mask` into each speaker position enum values to an array.
    pub fn channel_mask_to_speaker_positions(channel_mask: u32) -> Vec<u32> {
        let enums = [
            FRONT_LEFT,
            FRONT_RIGHT,
            FRONT_CENTER,
            LOW_FREQ,
            BACK_LEFT,
            BACK_RIGHT,
            FRONT_LEFT_OF_CENTER,
            FRONT_RIGHT_OF_CENTER,
            BACK_CENTER,
            SIDE_LEFT,
            SIDE_RIGHT,
            TOP_CENTER,
            TOP_FRONT_LEFT,
            TOP_FRONT_CENTER,
            TOP_FRONT_RIGHT,
            TOP_BACK_LEFT,
            TOP_BACK_CENTER,
            TOP_BACK_RIGHT,
        ];
        let mut ret = Vec::<u32>::new();
        for (i, m) in enums.iter().enumerate() {
            let m = *m as u32;
            if channel_mask & m == m {
                ret.push(enums[i]);
            }
        }
        ret
    }

    /// * Break down `channel_mask` into each speaker position description string.
    pub fn channel_mask_to_speaker_positions_descs(channel_mask: u32) -> Vec<&'static str> {
        channel_mask_to_speaker_positions(channel_mask)
            .iter()
            .map(|e| channel_bit_to_string(*e))
            .collect()
    }

    /// * Guess the channel mask by the given channel number.
    pub fn guess_channel_mask(channels: u16) -> Result<u32, AudioError> {
        match channels {
            0 => Err(AudioError::GuessChannelMaskFailed(channels)),
            1 => Ok(MONO_LAYOUT),
            2 => Ok(STEREO_LAYOUT),
            6 => Ok(DOLBY_5_1_FRONT_BACK_LAYOUT),
            8 => Ok(DOLBY_7_1_LAYOUT),
            o => {
                let mut mask = 0;
                for i in 0..o {
                    let bit = 1 << i;
                    if bit > 0x20000 {
                        return Err(AudioError::GuessChannelMaskFailed(channels));
                    }
                    mask |= bit;
                }
                Ok(mask)
            }
        }
    }

    /// * Check if the channel mask matches the channel number.
    pub fn is_channel_mask_valid(channels: u16, channel_mask: u32) -> bool {
        if channels <= 2 && channel_mask == 0 {
            return true;
        }
        let mut counter: u16 = 0;
        for i in 0..32 {
            if ((1 << i) & channel_mask) != 0 {
                counter += 1;
            }
        }
        counter == channels
    }
}

/// ## Downmixer params for dolby 5.1 or 7.1
#[derive(Debug, Clone, Copy)]
pub struct DownmixerParams {
    /// * Front left/right dB modifier
    pub front_lr_db: f64,

    /// * Front center dB modifier
    pub front_center_db: f64,

    /// * LFE dB modifier
    pub lowfreq_db: f64,

    /// * Side left/right dB modifier
    pub side_lr_db: f64,

    /// * Back left/right dB modifier
    pub back_lr_db: f64,
}

impl DownmixerParams {
    pub fn new() -> Self {
        Self {
            front_lr_db: 0.0,
            front_center_db: -3.0,
            lowfreq_db: -6.0,
            side_lr_db: -3.0,
            back_lr_db: -3.0,
        }
    }
}

impl Default for DownmixerParams {
    fn default() -> Self {
        Self::new()
    }
}

/// ## Downmixer to downmix multi-channels audio to stereo
#[derive(Debug, Clone, Copy)]
pub struct Downmixer {
    /// Num channels
    pub channels: u16,

    /// The channel mask indicates which channel is for which speaker.
    pub channel_mask: u32,

    /// The weights for downmixing
    pub gains: CopiableBuffer<f64, 8>,
}

impl Downmixer {
    fn normalize_gains(&mut self) {
        let sum: f64 = self.gains.iter().sum();
        self.gains = self.gains.iter().map(|x| x / sum).collect();
    }

    pub fn new(channel_mask: u32, params: DownmixerParams) -> Result<Self, AudioError> {
        let mut ret = Self {
            channels: 0,
            channel_mask,
            gains: CopiableBuffer::new(),
        };
        match channel_mask {
            SpeakerPosition::Dolby5_1LayoutFrontBack => {
                ret.channels = 6;

                // Front left
                ret.gains.push(db_to_gain(params.front_lr_db));

                // Front right
                ret.gains.push(db_to_gain(params.front_lr_db));

                // Front center
                ret.gains.push(db_to_gain(params.front_center_db));

                // Low freq
                ret.gains.push(db_to_gain(params.lowfreq_db));

                // Back left
                ret.gains.push(db_to_gain(params.back_lr_db));

                // Back right
                ret.gains.push(db_to_gain(params.back_lr_db));

                // Normalize the gains
                ret.normalize_gains();
                Ok(ret)
            }
            SpeakerPosition::Dolby5_1LayoutFrontSide => {
                ret.channels = 6;

                // Front left
                ret.gains.push(db_to_gain(params.front_lr_db));

                // Front right
                ret.gains.push(db_to_gain(params.front_lr_db));

                // Front center
                ret.gains.push(db_to_gain(params.front_center_db));

                // Low freq
                ret.gains.push(db_to_gain(params.lowfreq_db));

                // Side left
                ret.gains.push(db_to_gain(params.side_lr_db));

                // Side right
                ret.gains.push(db_to_gain(params.side_lr_db));

                // Normalize the gains
                ret.normalize_gains();
                Ok(ret)
            }
            SpeakerPosition::Dolby7_1Layout => {
                ret.channels = 8;

                // Front left
                ret.gains.push(db_to_gain(params.front_lr_db));

                // Front right
                ret.gains.push(db_to_gain(params.front_lr_db));

                // Front center
                ret.gains.push(db_to_gain(params.front_center_db));

                // Low freq
                ret.gains.push(db_to_gain(params.lowfreq_db));

                // Back left
                ret.gains.push(db_to_gain(params.side_lr_db));

                // Back right
                ret.gains.push(db_to_gain(params.side_lr_db));

                // Side left
                ret.gains.push(db_to_gain(params.side_lr_db));

                // Side right
                ret.gains.push(db_to_gain(params.side_lr_db));

                // Normalize the gains
                ret.normalize_gains();
                Ok(ret)
            }
            o => Err(AudioError::InvalidArguments(format!(
                "The input channel mask is not downmixable, it is {}",
                SpeakerPosition::channel_mask_to_string(o)
            ))),
        }
    }

    fn downmix_frame_to_stereo<S>(&self, frame: &[S]) -> (S, S)
    where
        S: SampleType {
        let gained: Vec<S> = frame.iter().enumerate().map(|(i, s)|S::cast_from(s.as_f64() * self.gains[i])).collect();
        match self.channel_mask {
            SpeakerPosition::Dolby5_1LayoutFrontBack | SpeakerPosition::Dolby5_1LayoutFrontSide=> {
                (
                    gained[0] + gained[2] + gained[3] + gained[4],
                    gained[1] + gained[2] + gained[3] + gained[5],
                )
            }
            SpeakerPosition::Dolby7_1Layout => {
                (
                    gained[0] + gained[2] + gained[3] + gained[4] + gained[6],
                    gained[1] + gained[2] + gained[3] + gained[5] + gained[7],
                )
            }
            o => panic!(
                "The input channel mask is not downmixable, it is {}",
                SpeakerPosition::channel_mask_to_string(o)
            ),
        }
    }

    pub fn downmix_frame_to_stereos<S>(&self, channel_mask: u32, frames: &[Vec<S>]) -> Result<Vec<(S, S)>, AudioError>
    where
        S: SampleType {
        if self.channel_mask != channel_mask {
            Err(AudioError::ChannekMaskNotMatch(format!(
            "The given `channel_mask` 0x{:x} does not match the `channel_mask` 0x{:x} when the downmixer was initialized",
            channel_mask, self.channel_mask
            )))
        } else {
            Ok(frames.iter().map(|frame|self.downmix_frame_to_stereo(frame)).collect())
        }
    }
}
