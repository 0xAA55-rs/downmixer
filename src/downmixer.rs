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
pub struct SpeakerPosition;

impl SpeakerPosition {
    pub const FrontLeft: u32 = 0x1;
    pub const FrontRight: u32 = 0x2;
    pub const FrontCenter: u32 = 0x4;
    pub const LowFreq: u32 = 0x8;
    pub const BackLeft: u32 = 0x10;
    pub const BackRight: u32 = 0x20;
    pub const FrontLeftOfCenter: u32 = 0x40;
    pub const FrontRightOfCenter: u32 = 0x80;
    pub const BackCenter: u32 = 0x100;
    pub const SideLeft: u32 = 0x200;
    pub const SideRight: u32 = 0x400;
    pub const TopCenter: u32 = 0x800;
    pub const TopFrontLeft: u32 = 0x1000;
    pub const TopFrontCenter: u32 = 0x2000;
    pub const TopFrontRight: u32 = 0x4000;
    pub const TopBackLeft: u32 = 0x8000;
    pub const TopBackCenter: u32 = 0x10000;
    pub const TopBackRight: u32 = 0x20000;

    pub fn channel_bit_to_string(bit: u32) -> &'static str {
        match bit {
            Self::FrontLeft => "front_left",
            Self::FrontRight => "front_right",
            Self::FrontCenter => "front_center",
            Self::LowFreq => "low_freq",
            Self::BackLeft => "back_left",
            Self::BackRight => "back_right",
            Self::FrontLeftOfCenter => "front_left_of_center",
            Self::FrontRightOfCenter => "front_right_of_center",
            Self::BackCenter => "back_center",
            Self::SideLeft => "side_left",
            Self::SideRight => "side_right",
            Self::TopCenter => "top_center",
            Self::TopFrontLeft => "top_front_left",
            Self::TopFrontCenter => "top_front_center",
            Self::TopFrontRight => "top_front_right",
            Self::TopBackLeft => "top_back_left",
            Self::TopBackCenter => "top_back_center",
            Self::TopBackRight => "top_back_right",
            _ => "Invalid bit",
        }
    }

    pub fn channel_mask_to_string(channel_mask: u32) -> String {
        Self::channel_mask_to_speaker_positions_descs(channel_mask).join(" + ")
    }

    /// * Break down `channel_mask` into each speaker position enum values to an array.
    pub fn channel_mask_to_speaker_positions(channel_mask: u32) -> Vec<u32> {
        let enums = [
            Self::FrontLeft,
            Self::FrontRight,
            Self::FrontCenter,
            Self::LowFreq,
            Self::BackLeft,
            Self::BackRight,
            Self::FrontLeftOfCenter,
            Self::FrontRightOfCenter,
            Self::BackCenter,
            Self::SideLeft,
            Self::SideRight,
            Self::TopCenter,
            Self::TopFrontLeft,
            Self::TopFrontCenter,
            Self::TopFrontRight,
            Self::TopBackLeft,
            Self::TopBackCenter,
            Self::TopBackRight,
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
        Self::channel_mask_to_speaker_positions(channel_mask)
            .iter()
            .map(|e| Self::channel_bit_to_string(*e))
            .collect()
    }

    /// * Guess the channel mask by the given channel number.
    pub fn guess_channel_mask(channels: u16) -> Result<u32, AudioError> {
        match channels {
            0 => Err(AudioError::GuessChannelMaskFailed(channels)),
            1 => Ok(Self::MonoLayout),
            2 => Ok(Self::StereoLayout),
            6 => Ok(Self::Dolby5_1LayoutFrontBack),
            8 => Ok(Self::Dolby7_1Layout),
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

    /// * The channel mask for mono audio layout
    pub const MonoLayout: u32 = Self::FrontCenter;

    /// * The channel mask for stereo audio layout
    pub const StereoLayout: u32 = Self::FrontLeft | Self::FrontRight;

    /// * The channel mask for dolby 5.1 audio layout
    pub const Dolby5_1LayoutFrontBack: u32 = Self::FrontLeft
        | Self::FrontRight
        | Self::FrontCenter
        | Self::BackLeft
        | Self::BackRight
        | Self::LowFreq;
    pub const Dolby5_1LayoutFrontSide: u32 = Self::FrontLeft
        | Self::FrontRight
        | Self::FrontCenter
        | Self::SideLeft
        | Self::SideRight
        | Self::LowFreq;

    /// * The channel mask for dolby 7.1 audio layout
    pub const Dolby7_1Layout: u32 = Self::FrontLeft
        | Self::FrontRight
        | Self::FrontCenter
        | Self::SideLeft
        | Self::SideRight
        | Self::BackLeft
        | Self::BackRight
        | Self::LowFreq;
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
                "The input channel mask is not dolby 5.1 layout, it is {}",
                SpeakerPosition::channel_mask_to_string(o)
            ))),
        }
    }
}
