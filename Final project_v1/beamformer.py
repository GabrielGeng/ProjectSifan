import math
from enum import Enum
import numpy as np
from base.geometry import incident_angle_to_dir, incident_dir_to_angle, Angle3D
from base.math import qr_linear_solve, cartesian_product, svd_pinv, detect_peak, get_envelop, complex_normalize
from scipy.linalg import hadamard, toeplitz
from beamforming.converter import TimeFreqDomainConverter, TimeFreqConverterMode
from utils.frame_data import SingleChanFrameData, MultiChanFrameData
from beamforming.vad_doa_processor import VadDoaProcessor
from beamforming.post_filter import MultiChanPreFilter, McraNoiseEstimatorConfig, McraNoiseEstimator
from utils.data_conversion import save_wave_data, split_frames_2d, split_frames
from base.microphone_array import LinearMicrophoneArray, CircularMicrophoneArray
from base.data_helper import IterativeLinearSmoother
from DOA.DOA_estimator import FreqLmsDoaEstimator, GccPhatDoaEstimator
from base.signal import pre_emphasis, de_emphasis, get_mel_filter_bank_pair


# Abstract class for all beamformers
class Beamformer:
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed,
                 vad_doa_process=False, vad_doa_processor_config=None, debug=False):
        self.mic_array = mic_array
        self.sound_speed = sound_speed
        self.sample_rate = sample_rate
        self.tau = np.zeros(self.mic_array.num_mic)
        self.vad_doa_processor = None
        self._debug = debug
        if vad_doa_process:
            self.vad_doa_processor = VadDoaProcessor(mic_array, incident_angle, sample_rate, sound_speed,
                                                     vad_doa_processor_config)
        self.update_incident_angle(incident_angle)

    def process(self, data, incident_angle=None):
        raise NotImplementedError

    def update_incident_angle(self, incident_angle):
        self.tau = self._get_tau_from_angle(incident_angle)
        if self.vad_doa_processor is not None:
            self.vad_doa_processor.reset_incident_angle(incident_angle)

    def _get_tau_from_angle(self, angle):
        incident_dir = incident_angle_to_dir(angle)
        tau = -np.array([np.matmul(incident_dir.T, mic_pos.to_numpy_array()) / self.sound_speed
                        for mic_pos in self.mic_array.mic_coordinates])
        return tau

    def _steer(self, data):
        delay_samples = -self.tau * self.sample_rate
        assert data.shape[0] == delay_samples.shape[0]
        steered_data = np.zeros_like(data)
        data_length = data.shape[1]
        for index, delay in enumerate(delay_samples):
            assert -data_length < delay < data_length
            int_delay = int(round(delay))
            frac_delay = delay - int_delay
            # Using freq method to generate fractional delays
            if frac_delay != 0:
                nfft = int(math.pow(2, math.ceil(math.log2(data_length + max(0, int_delay)))))
                fft_bin = 2 * math.pi * (np.arange(0, nfft / 2 + 1)) / nfft
                spec = np.fft.rfft(data[index], nfft)
                delayed_data = np.fft.irfft(spec * np.exp(-1j * delay * fft_bin))
                if int_delay > 0:
                    delayed_data[:int_delay] = 0
                steered_data[index] = delayed_data[0:data_length]
            # Sample operation for integer delays
            elif int_delay > 0:
                steered_data[index, int_delay:] = data[index, 0:-int_delay]
            elif int_delay < 0:
                steered_data[index, 0:int_delay] = data[index, -int_delay:]
            else:
                steered_data[index] = data[index]
        return steered_data

    @staticmethod
    # Calculate LCMV weights
    # x: array with size [m, N], where m is degree of freedom of the beamformer,
    #   which equals to number_of_channels * filter_length, N is number of samples
    # constraints: array with size [k, m], where k is number of constraints, m is number of channels
    # responses: array with size [k], where k is number of constraints
    # diagonal_loading: scalar value
    def _lcmv_weights(x, constraints, responses, diagonal_loading):
        # Number of samples should be at least equal to the number of channels
        assert(x.shape[1] >= x.shape[0])
        x = x / math.sqrt(x.shape[1])
        if diagonal_loading != 0:
            x = np.concatenate((x, np.eye(x.shape[0]) * math.sqrt(diagonal_loading)), axis=1)
        if len(constraints.shape) == 1:
            # MVDR case: w = inv(S) * a / (a.H * inv(S) * a)
            [temp, _] = qr_linear_solve(x, constraints.T)
            w = responses * temp / np.matmul(constraints.T.conj(), temp)
        else:
            # LCMV case: w = inv(S) * C * inv(C.H * inv(S) * C) * R
            [temp_1, f] = qr_linear_solve(x, constraints.T)
            [temp_2, _] = qr_linear_solve(f.T.conj(), responses)
            w = np.matmul(temp_1, temp_2)
        return w


# Abstract class for all frequency domain beamformers
class AbstractFreqBeamformer(Beamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed, fft_length=64, operating_freq=0,
                 vad_doa_process=False, vad_doa_processor_config=None, debug=False):
        self._time_freq_converter = TimeFreqDomainConverter(fft_length,
                                                            mode=TimeFreqConverterMode.overlap_add_sqrt_hamm_window)
        self._fft_length = self._time_freq_converter.fft_length
        self.freq_dim = self._time_freq_converter.rfft_length
        self._subband_freqs = np.linspace(0, sample_rate / 2, self.freq_dim) + operating_freq
        self._steer_vectors = None
        self._steer_vectors_uniform_gain = None
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed,
                         vad_doa_process=vad_doa_process, vad_doa_processor_config=vad_doa_processor_config,
                         debug=debug)
        self.update_incident_angle(incident_angle)

    def update_incident_angle(self, incident_angle):
        super(AbstractFreqBeamformer, self).update_incident_angle(incident_angle)
        self._steer_vectors = self._get_steer_vectors_from_tau(self.tau)
        self._steer_vectors_uniform_gain = self._steer_vectors

    def _get_steer_vectors_from_tau(self, tau):
        tau_temp = np.zeros(self.mic_array.num_mic)
        if tau.shape == tau_temp.shape:
            tau_temp = tau
        elif tau.shape[0] == tau_temp.shape[0] - 1:
            tau_temp[1:] = tau
            tau_temp[0] = 0
        else:
            raise Exception("Shape of tau is incorrect!")
        tau_temp = np.tile(tau_temp, [self.freq_dim, 1])
        subband_freqs_temp = np.tile(self._subband_freqs, [self.mic_array.num_mic, 1]).T
        steer_vectors = np.exp(-1j * 2 * np.pi * tau_temp * subband_freqs_temp)
        return steer_vectors

    def _update_channel_gain(self, channel_gain):
        self._steer_vectors = self._steer_vectors_uniform_gain * channel_gain

    def _subband_fft(self, x):
        return self._time_freq_converter.convert_to_freq_domain(x)

    def _subband_ifft(self, yf):
        return self._time_freq_converter.convert_to_time_domain(yf)

    def _constrain_weight_norm(self, weight, threshold=10.0, frame_index=0):
        weight_norm = np.real(weight * weight.conj())
        avg_amp = np.mean(np.sqrt(weight_norm))
        if self._debug:
            print("time %f avg_amp: %f" %
                  (frame_index * self._time_freq_converter.interval / self.sample_rate, avg_amp))
        sum_weight_norm = np.sum(weight_norm, axis=0)
        weight = np.where(sum_weight_norm > threshold, np.sqrt(threshold / sum_weight_norm) * weight, weight)
        return weight

    def process(self, data, incident_angle=None):
        pass


# Implements a conventional, time-delay based beamformer
class TimeDelayBeamformer(Beamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed,
                 vad_doa_process=False, vad_doa_processor_config=None):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed,
                         vad_doa_process=vad_doa_process, vad_doa_processor_config=vad_doa_processor_config)
        self.weights = np.array([1 / self.mic_array.num_mic] * self.mic_array.num_mic)

    def process(self, data, incident_angle=None):
        if incident_angle is not None:
            self.update_incident_angle(incident_angle)
        if self.vad_doa_processor is not None:
            self.vad_doa_processor.doa_process(data)
        beam = np.matmul(self.weights, self._steer(data))
        if self.vad_doa_processor is not None:
            beam = self.vad_doa_processor.filter_process(beam)
        return beam


# Implements a wide-band time domain LCMV(Least Constrained Minimum Variance) beamformer
class LcmvBeamformer(Beamformer):
    # filter_length: the filter length for each channel of the beamformer
    # constraints: array with size [k, m], where k is number of constraints, m is degree of freedom of the beamformer,
    #   which equals to number_of_channels * filter_length
    # responses: array with size [k], where k is number of constraints
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed, filter_length, constraints, responses,
                 diagonal_loading=0):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed)
        self.weights = None
        assert(filter_length > 0)
        self.degree_of_freedom = filter_length * mic_array.num_mic
        if len(constraints.shape) == 1:
            assert(responses.shape[0] == 1)
            assert(constraints.shape[0] == self.degree_of_freedom)
        else:
            assert(constraints.shape[0] == responses.shape[0])
            assert(constraints.shape[1] == self.degree_of_freedom)
        self.constraints = constraints
        self.responses = responses
        self.filter_length = filter_length
        self.diagonal_loading = diagonal_loading
        if self.filter_length > 1:
            self.data_buffer = np.zeros([mic_array.num_mic, self.filter_length - 1])
        else:
            self.data_buffer = None

    def process(self, data, incident_angle=None, train_data=None):
        if incident_angle is not None:
            self.update_incident_angle(incident_angle)
        data_steered = self._steer(data)
        data_steered = self._prepare_space_time_snapshot(data_steered)
        if train_data is not None:
            train_data_steered = self._steer(train_data)
            train_data_steered = self._prepare_space_time_snapshot(train_data_steered)
            self.weights = self._lcmv_weights(train_data_steered, self.constraints, self.responses,
                                              self.diagonal_loading)
        elif self.weights is None:
            self.weights = self._lcmv_weights(data_steered, self.constraints, self.responses, self.diagonal_loading)
        return np.matmul(self.weights.conj(), data_steered)

    def _prepare_space_time_snapshot(self, x):
        if self.data_buffer is None:
            return x
        else:
            buffer_length = self.data_buffer.shape[1]
            indices_1 = np.arange(self.mic_array.num_mic)
            indices_1 = np.tile(indices_1, (x.shape[1], 1)).T
            indices_1 = np.tile(indices_1, (self.filter_length, 1))
            indices_2 = np.zeros((self.filter_length, x.shape[1]), dtype=np.int32)
            for i in range(indices_2.shape[0]):
                indices_2[i, :] = np.arange(x.shape[1]) + buffer_length - i
            indices_2 = np.repeat(indices_2, self.mic_array.num_mic, axis=0)
            x = np.concatenate((self.data_buffer, x), axis=1)
            self.data_buffer = x[:, -buffer_length:]
            space_time_snap = x[indices_1, indices_2]
            return space_time_snap


# Frost beamformer is a wide-band time domain implementation of the
# MVDR(Minimum Variance Distortion Response) beamformer
# It is also a special case of LCMV(Least Constrained Minimum Variance) beamformer
class FrostBeamformer(LcmvBeamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed, filter_length, diagonal_loading=0):
        constraints = np.eye(filter_length)
        constraints = np.repeat(constraints, mic_array.num_mic, axis=1)
        responses = np.zeros(filter_length)
        responses[0] = 1
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed, filter_length, constraints, responses,
                         diagonal_loading)


# Wide-band MVDR beamformer in frequency domain
class FreqMvdrBeamformer(AbstractFreqBeamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed,
                 fft_length=64, operating_freq=0, diagonal_loading=0):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed, fft_length, operating_freq)
        self.weights = None
        self.diagonal_loading = diagonal_loading

    def process(self, data, incident_angle=None, train_data=None):
        input_length = data.shape[1]
        if incident_angle is not None:
            self.update_incident_angle(incident_angle)
        xf = MultiChanFrameData(self._subband_fft(data))
        if train_data is not None:
            xtf = MultiChanFrameData(self._subband_fft(train_data))
            self.weights = self.__cal_weights(xtf)
        elif self.weights is None:
            self.weights = self.__cal_weights(xf)
        yf = SingleChanFrameData.initialize(xf.num_frames, xf.feature_dim, dtype=xf.dtype)
        for frame_index in range(yf.num_frames):
            xf_frame = xf.get_frame_data(frame_index)
            yf.set_frame_data(frame_index, np.sum(self.weights.conj() * xf_frame, axis=0))
        return self._subband_ifft(yf.get_data())[:input_length]

    def __cal_weights(self, xf):
        weights = np.zeros((self.mic_array.num_mic, self.freq_dim), dtype=np.complex64)
        for i in range(self.freq_dim):
            weights[:, i] = self._lcmv_weights(xf.get_feature_data(i),
                                               self._steer_vectors[i, :], 1, self.diagonal_loading)
        return weights


# Iterative Frost beamformer in time domain
class IterativeFrostBeamformer(FrostBeamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed, filter_length, diagonal_loading=0,
                 frame_length=64, learning_rate=0.1):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed, filter_length, diagonal_loading)
        self.frame_length = frame_length
        self.frame_interval = self.frame_length >> 1
        self.learning_rate = learning_rate
        self.p = None
        self.f = None
        self.window = np.linspace(0, 1, self.frame_length >> 1)
        self.window = np.concatenate((self.window, np.flipud(self.window)))
        self.__update_p_and_f()

    def process(self, data, incident_angle=None, train_data=None):
        if incident_angle is not None:
            self.update_incident_angle(incident_angle)
            self.__update_p_and_f()
        data_steered = self._steer(data)
        data_steered = self._prepare_space_time_snapshot(data_steered)
        num_frames = 1 + int(np.ceil((data_steered.shape[1] - self.frame_length) / self.frame_interval))
        y = np.zeros(data_steered.shape[1])
        self.weights = self.f
        # Loop through all frames
        for i in range(num_frames):
            frame_start = i * self.frame_interval
            frame_end = min(frame_start + self.frame_length, data_steered.shape[1])
            x_frame = data_steered[:, frame_start:frame_end]
            y_frame = np.matmul(self.weights.conj(), x_frame)
            # w <- F + P(w - u * x * y)
            weights_update = self.weights - self.learning_rate * np.sum(x_frame * y_frame, axis=1)
            self.weights = self.f + np.matmul(self.p, weights_update)
            y[frame_start:frame_end] += y_frame * self.window[:frame_end - frame_start]
        return y

    def __update_p_and_f(self):
        chc = np.matmul(self.constraints.conj(), self.constraints.T)
        inv_chc = np.linalg.inv(chc)
        c_inv_chc = np.matmul(self.constraints.T, inv_chc)
        # F = C * inv(C.H * C) * R
        self.f = np.matmul(c_inv_chc, self.responses)
        # P = I - C * inv(C.H * C) * C.H
        self.p = np.eye(self.degree_of_freedom) - np.matmul(c_inv_chc, self.constraints.conj())
        pass


# Iterative wide-band MVDR beamformer in frequency domain
class IterativeFreqMvdrBeamformer(AbstractFreqBeamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed,
                 fft_length=64, operating_freq=0, learning_rate=0.1, update_interference=False,
                 vad_doa_process=False, vad_doa_processor_config=None):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed, fft_length, operating_freq,
                         vad_doa_process=vad_doa_process, vad_doa_processor_config=vad_doa_processor_config)
        self.weights = None
        self.p = None
        self.f = None
        self.learning_rate = learning_rate
        self.p_est = IterativeLinearSmoother(0.9)
        self.__update_interference = update_interference
        self.__interference_steer_vectors = None
        self.__update_p_and_f()
        tau_update_time = 0.5
        self.__tau_update_frames = int(tau_update_time * self.sample_rate / self._time_freq_converter.interval)
        self.__last_tau_update = 0

    def process(self, data, incident_angle=None):
        if incident_angle is not None:
            self.update_incident_angle(incident_angle)
            self.__update_p_and_f()
        if self.vad_doa_processor is not None:
            self.vad_doa_processor.doa_process(data)
        input_length = data.shape[1]
        self.weights = self.f
        xf = MultiChanFrameData(self._subband_fft(data))
        yf = SingleChanFrameData.initialize(xf.num_frames, xf.feature_dim, dtype=xf.dtype)
        # Loop through frames
        for frame_index in range(yf.num_frames):
            if self.vad_doa_processor is not None:
                self.__tau_update_process(frame_index)
            xf_frame = xf.get_frame_data(frame_index)
            # yf = w.H * xf
            yf.set_frame_data(frame_index, np.sum(self.weights.conj() * xf_frame, axis=0))
            # power estimation
            p = np.sum(np.real(xf_frame * xf_frame.conj()), axis=0)
            self.p_est.update(p)
            # w <- P * (w - u * xf * yf.H) + F
            update = self.weights -\
                     self.learning_rate * xf_frame * yf.get_frame_data(frame_index).conj() / self.p_est.data
            for freq_index in range(update.shape[1]):
                update[:, freq_index] = np.matmul(self.p[freq_index, :, :], update[:, freq_index])
            self.weights = update + self.f
            self.weights = self._constrain_weight_norm(self.weights, threshold=10, frame_index=frame_index)
        beam = self._subband_ifft(yf.get_data())[:input_length]
        if self.vad_doa_processor is not None:
            beam = self.vad_doa_processor.filter_process(beam)
        return beam

    def __tau_update_process(self, frame_index):
        if frame_index - self.__last_tau_update == self.__tau_update_frames:
            start_sample_index = (self.__last_tau_update + 1) * self._time_freq_converter.interval
            end_sample_index = frame_index * self._time_freq_converter.interval + self._time_freq_converter.fft_length
            tau, interference_tau = self.vad_doa_processor.get_tau(start_sample_index, end_sample_index)
            update = False
            if tau is not None:
                # print("frame %d tau updated to:" % frame_index)
                # print(tau)
                self._steer_vectors = self._get_steer_vectors_from_tau(tau)
                update = True
            if interference_tau is not None and self.__update_interference:
                self.__interference_steer_vectors = self._get_steer_vectors_from_tau(interference_tau)
                update = True
            elif self.__interference_steer_vectors is not None:
                update = True
                self.__interference_steer_vectors = None
            # print("frame: %d time: %f interference tau updated to:" %
            #   (frame_index, frame_index * self._time_freq_converter.interval / self.sample_rate))
            # print(interference_tau)
            if update:
                self.__update_p_and_f()
            self.__last_tau_update = frame_index

    def __update_p_and_f(self):
        self.p = np.zeros((self._steer_vectors.shape[0], self._steer_vectors.shape[1], self._steer_vectors.shape[1]),
                          dtype=np.complex64)
        # MVDR case
        if self.__interference_steer_vectors is None:
            norm = np.sum(self._steer_vectors.conj() * self._steer_vectors, axis=1)
            norm = np.tile(norm, [self.mic_array.num_mic, 1])
            # F = a / (|a|^2)
            self.f = self._steer_vectors.T / norm
            # P = I - a * a.H / (|a|^2)
            for i in range(self.p.shape[0]):
                a = self._steer_vectors[i, :]
                self.p[i, :, :] = np.eye(self.p.shape[1]) - cartesian_product(a, a.conj()) / norm[:, i]
        # LCMV case
        else:
            self.f = np.zeros_like(self._steer_vectors.T)
            for i in range(self._steer_vectors.shape[0]):
                c = np.array([self._steer_vectors[i], self.__interference_steer_vectors[i]]).T
                chc = np.matmul(c.T.conj(), c)
                if np.linalg.matrix_rank(chc) < 2 or np.linalg.cond(chc, 1) > 100:
                    a = c[:, 0]
                    self.f[:, i] = a / chc[0, 0]
                    self.p[i, :, :] = np.eye(self.p.shape[1]) - cartesian_product(a, a.conj()) / chc[0, 0]
                else:
                    r = np.array([1, 0.01])
                    # F = C * inv(C.H * C) * R
                    temp = np.linalg.solve(chc, r)
                    self.f[:, i] = np.matmul(c, temp)
                    # P = I - C * inv(C.H * C) * C.H
                    temp = np.linalg.solve(chc, c.T.conj())
                    self.p[i, :, :] = np.eye(self.p.shape[1]) - np.matmul(c, temp)


# GSC(General Sidelobe Canceller) beamformer
# A GSC beamformer splits the incoming signals into two channels.
# One channel goes through a conventional beamformer path and the second goes into a sidelobe canceling path.
# The algorithm first pre-steers the array to the beamforming direction and then adaptively chooses filter weights
# to minimize power at the output of the sidelobe canceling path.
# The algorithm uses least mean squares (LMS) to compute the adaptive weights.
# The final beamformed signal is the difference between the outputs of the two paths.
class GscBeamformer(Beamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed, filter_length,
                 frame_length=64, learning_rate=0.1):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed)
        self.conv_weights = np.array([1 / self.mic_array.num_mic] * self.mic_array.num_mic)
        self.filter_length = filter_length
        assert frame_length % 2 == 0
        self.frame_length = frame_length
        self.frame_interval = self.frame_length >> 1
        self.__initialize_block_matrix()
        self.sc_weights = np.ones((self.block_matrix.shape[0], self.filter_length)) / self.block_matrix.shape[0]
        self.window = np.linspace(0, 1, self.frame_length >> 1)
        self.window = np.concatenate((self.window, np.flipud(self.window)))
        self.learning_rate = learning_rate
        if self.filter_length > 1:
            self.data_buffer = np.zeros([mic_array.num_mic, self.filter_length - 1])
        else:
            self.data_buffer = None

    def process(self, data, incident_angle=None):
        buffer_length = self.data_buffer.shape[1]

        # Conventional path
        if incident_angle is not None:
            self.update_incident_angle(incident_angle)
        if self.data_buffer is not None:
            data = np.concatenate((self.data_buffer, data), axis=1)
        data_steered = self._steer(data)
        y_conv = np.matmul(self.conv_weights, data_steered)
        if self.data_buffer is not None:
            y_conv = y_conv[buffer_length:]
            self.data_buffer = data[:, -buffer_length:]

        # Sidelobe Cancelling path
        num_frames = 1 + int(np.ceil((y_conv.shape[0] - self.frame_length) / self.frame_interval))
        y_frame = np.zeros(self.frame_length)
        y = np.zeros_like(y_conv)
        # Loop through all frames
        for i in range(num_frames):
            frame_start = i * self.frame_interval
            frame_end = min(frame_start + self.frame_length, y_conv.shape[0])
            weights_update = np.zeros_like(self.sc_weights)
            # Loop through samples in a frame, not updating weights
            for j in range(frame_start, frame_end):
                data_null = np.matmul(self.block_matrix, data_steered[:, j:j+self.filter_length])
                p_null = np.sum(data_null * data_null)
                y_frame[j - frame_start] = error = y_conv[j] - np.sum(self.sc_weights * data_null)
                # w <- w + u * y.h * u / P(u)
                weights_update += self.learning_rate / p_null * np.conj(error) * data_null
            # Update weights after a frame has been processed
            self.sc_weights = self.sc_weights + weights_update
            y[frame_start:frame_end] += (y_frame * self.window)[:frame_end - frame_start]
        return y

    def __initialize_block_matrix(self):
        if int(math.log2(self.mic_array.num_mic)) == math.log2(self.mic_array.num_mic):
            self.block_matrix = hadamard(self.mic_array.num_mic)[1:, :]
        else:
            c = np.zeros(self.mic_array.num_mic-1)
            c[0] = 1
            r = np.zeros(self.mic_array.num_mic)
            r[0] = 1
            r[1] = -1
            self.block_matrix = toeplitz(c, r)


# TF-GSC(Transfer Function General Sidelobe Canceller) Beamformer
class TfGscBeamformer(AbstractFreqBeamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed, filter_length=-1,
                 fft_length=64, operating_freq=0, learning_rate=0.1,
                 vad_doa_process=False, vad_doa_processor_config=None, adaptive_tf=False):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed, fft_length, operating_freq,
                         vad_doa_process=vad_doa_process, vad_doa_processor_config=vad_doa_processor_config)
        self.filter_length = filter_length
        self.conv_weights = None
        self.sc_weights = None
        self.block_matrix = None
        self.learning_rate = learning_rate
        self.p_est = IterativeLinearSmoother(0.9)
        self.__f = None
        self.__update_f()
        self.__incident_angle = incident_angle
        self.reset_incident_angle(incident_angle)
        self.adaptive_tf = adaptive_tf
        # TF update time in seconds
        tf_update_time = 0.2
        # TF buffer length in seconds
        tf_buffer_length = 0.5
        self.tf_update_frames = int(np.ceil(tf_update_time * self.sample_rate / self._time_freq_converter.interval))
        self.tf_buffer_frames = int(np.ceil(tf_buffer_length * self.sample_rate / self._time_freq_converter.interval))
        self.xf_buffer = None
        self.xf_buffer_index = 0
        self.tf_update_count = self.tf_update_frames
        # constants used to check RIR validity
        self.mic_pairs = [(0, i + 1) for i in range(mic_array.num_mic - 1)]
        self.mic_pos_diffs = np.array([self.mic_array.mic_coordinates[pair[1]].to_numpy_array() -
                                       self.mic_array.mic_coordinates[pair[0]].to_numpy_array()
                                       for pair in self.mic_pairs])
        self.mic_distances = [np.linalg.norm(pos_diff, 2) for pos_diff in self.mic_pos_diffs]
        self.max_delay = int(math.ceil(max(self.mic_distances) / self.sound_speed * self.sample_rate))
        self.pinv_factor = svd_pinv(self.mic_pos_diffs)

    def process(self, data, incident_angle=None):
        if incident_angle is not None:
            self.reset_incident_angle(incident_angle)
        if self.vad_doa_processor is not None:
            self.vad_doa_processor.doa_process(data)
        input_length = data.shape[1]
        self.sc_weights = np.ones((self.block_matrix.shape[1], self.freq_dim), dtype=np.complex64)\
                          / self.block_matrix.shape[1]
        xf = MultiChanFrameData(self._subband_fft(data))
        yfbf = SingleChanFrameData.initialize(xf.num_frames, xf.feature_dim, dtype=xf.dtype)
        yf = SingleChanFrameData.initialize(xf.num_frames, xf.feature_dim, dtype=xf.dtype)
        uf_all = np.zeros((self.block_matrix.shape[1], xf.num_frames, xf.feature_dim), dtype=xf.dtype)
        debug_info = np.zeros((4 * (self.mic_array.num_mic + 1), yf.num_frames, yf.feature_dim))
        # Sidelobe cancelling path, loop through frames
        for frame_index in range(yf.num_frames):
            # Conventional path
            xf_frame = xf.get_frame_data(frame_index)
            if self.adaptive_tf:
                self.__tf_estimator_process(xf_frame, self.__incident_angle, frame_index)
            yfbf.set_frame_data(frame_index, np.sum(self.conv_weights.conj() * xf_frame, axis=0))
            # u = b * x
            u = np.zeros((self.block_matrix.shape[1], xf.feature_dim), dtype=np.complex64)
            for freq_index in range(xf.feature_dim):
                u[:, freq_index] = np.matmul(self.block_matrix[freq_index, :, :], xf_frame[:, freq_index])
            uf_all[:, frame_index, :] = u
            # y = yfbf - g * u
            y = yfbf.get_frame_data(frame_index) - np.sum(self.sc_weights.conj() * u, axis=0)
            # power estimation
            p = np.sum(np.real(xf_frame * xf_frame.conj()), axis=0)
            self.p_est.update(p)
            # g <- g + mu * u * y.H / p
            weights_update = u * y.conj() / self.p_est.data
            new_weights = self.sc_weights + self.learning_rate * weights_update
            self.sc_weights = self.__weights_fir_truncate(new_weights)
            self.sc_weights = self._constrain_weight_norm(self.sc_weights, threshold=50, frame_index=frame_index)
            yf.set_frame_data(frame_index, np.copy(y))
        beam = self._subband_ifft(yf.get_data())[:input_length]
        if self.vad_doa_processor is not None:
            beam = self.vad_doa_processor.filter_process(beam)
        u_all = np.zeros((self.block_matrix.shape[1], input_length))
        for i in range(u_all.shape[0]):
            u_all[i] = self._subband_ifft(uf_all[i])[:input_length]
        # save_wave_data("u.wav", (u_all * 32768).astype(np.int16))
        return beam

    def reset_incident_angle(self, incident_angle):
        self.__incident_angle = incident_angle
        self.update_incident_angle(incident_angle)
        self.__update_tf()

    def preprocess(self, data, get_u=False):
        xf = MultiChanFrameData(self._subband_fft(data))
        yfbf = SingleChanFrameData.initialize(xf.num_frames,
                                              xf.feature_dim, dtype=xf.dtype)
        for i in range(xf.num_frames):
            yfbf.set_frame_data(i, np.sum(self.conv_weights.conj() * xf.get_frame_data(i), axis=0))
        if get_u:
            data_out = MultiChanFrameData.initialize(xf.num_channels, xf.num_frames, xf.feature_dim,
                                                     dtype=xf.dtype)
            data_out.set_channel_data(0, yfbf.get_data())
            for freq_index in range(xf.feature_dim):
                data_out.set_feature_data(freq_index,
                                          np.matmul(self.block_matrix[freq_index, :, :],
                                                    xf.get_feature_data(freq_index)),
                                          channel_start=1)
            return data_out
        else:
            return yfbf

    def postprocess(self, yf):
        return self._subband_ifft(yf)

    def __tf_estimator_process(self, xf_frame, incident_angle, frame_index):
        if self.xf_buffer is None:
            self.xf_buffer = MultiChanFrameData.initialize(self.mic_array.num_mic,
                                                           self.tf_buffer_frames, self.freq_dim, dtype=xf_frame.dtype)
        self.xf_buffer.set_frame_data(self.xf_buffer_index, xf_frame)
        self.xf_buffer_index += 1
        if self.xf_buffer_index == self.xf_buffer.num_frames:
            self.xf_buffer_index = 0
        self.tf_update_count -= 1
        if self.tf_update_count == 0:
            # print("frame %d time %s" %
            # (frame_index, frame_index * self._time_freq_converter.interval / self.sample_rate))
            self.__update_tf(self.xf_buffer, incident_angle)
            self.tf_update_count = self.tf_update_frames

    def __update_f(self):
        tau = self._get_tau_from_angle(Angle3D(0, 0))
        max_tau_index = np.argmax(tau)
        min_tau_index = np.argmin(tau)
        steer_vec = self._get_steer_vectors_from_tau(tau)
        self.__f = steer_vec[:, max_tau_index] / steer_vec[:, min_tau_index]

    def __update_dtf(self, interference_angle=None):
        if interference_angle is None:
            self.__update_tf()
        else:
            a = self._steer_vectors
            interference_tau = self._get_tau_from_angle(interference_angle)
            b = self._get_steer_vectors_from_tau(interference_tau)
            self.__update_w0_bm_dtf(a, b)

    def __update_tf(self, xf=None, incident_angle=None):
        # If xf is None, update h using steer vector
        if xf is None:
            a0 = np.tile(self._steer_vectors[:, 0], (self.mic_array.num_mic, 1)).T
            h = self._steer_vectors / a0
            self.__update_w0_bm(h)
        # Else, estimate h from input data
        else:
            h = np.zeros((xf.feature_dim, xf.num_channels), dtype=xf.dtype)
            phi0 = np.zeros(xf.feature_dim, dtype=xf.dtype)
            denominator = np.copy(phi0)
            for i in range(xf.num_channels):
                phi = xf.get_channel_data(i) * xf.get_channel_data(0).conj()
                if i == 0:
                    phi0 = phi
                numerator = np.mean(phi0 * phi, axis=0) - np.mean(phi0, axis=0) * np.mean(phi, axis=0)
                if i == 0:
                    denominator = np.where(np.abs(numerator) > 1e-10, numerator, 1e-10)
                    h[:, i] = 1
                else:
                    h[:, i] = numerator / denominator
            # rxxf = np.zeros((xf.num_channels, xf.num_channels), dtype=xf.dtype)
            # for i in range(xf.num_frames):
            #     rxxf += np.matmul(xf.get_frame_data(i), xf.get_frame_data(i).conj().T)
            # rxxf /= xf.num_frames
            # print(np.abs(np.linalg.eigvals(rxxf)))
            if self.__check_tf_validity(h, incident_angle):
                # print("tf updated!")
                self.__update_w0_bm(h)

    def __update_w0_bm(self, h):
        norm = np.sum(h.conj() * h, axis=1)
        norm = np.tile(norm, [self.mic_array.num_mic, 1])
        f = np.tile(self.__f.conj(), [self.mic_array.num_mic, 1])
        # w0 = h / (|h|^2) * f
        self.conv_weights = h.T / norm * f
        self.block_matrix = np.zeros((self._steer_vectors.shape[0], self.mic_array.num_mic - 1, self.mic_array.num_mic),
                                     dtype=np.complex64)
        for i in range(self.block_matrix.shape[0]):
            self.block_matrix[i, :, :] = np.concatenate((-h[i:i+1, 1:], np.eye(self.mic_array.num_mic - 1)), axis=0).T

    def __update_w0_bm_dtf(self, a, b):
        a_square = a.conj() * a
        b_square = b.conj() * b
        a_norm = np.sum(a_square, axis=1)
        b_norm = np.sum(b_square, axis=1)
        self.conv_weights = np.zeros_like(a.T)
        self.block_matrix = np.zeros((self._steer_vectors.shape[0], self.mic_array.num_mic - 2, self.mic_array.num_mic),
                                     dtype=np.complex64)
        for i in range(self.conv_weights.shape[1]):
            # conv weights
            denominator = a_norm[i] * b_norm[i] - np.matmul(a[i].conj(), b[i]) * np.matmul(b[i].conj(), a[i])
            bad_condition = np.abs(denominator) < 0.01
            if bad_condition:
                h = a[i] / a[i, 0]
                norm = np.sum(h * h.conj())
                self.conv_weights[:, i] = h / norm * self.__f[i].conj()
                self.block_matrix[i, :, :] = np.concatenate((np.array([-h[2:], np.zeros(self.mic_array.num_mic - 2)]),
                                                             np.eye(self.mic_array.num_mic - 2)), axis=0).T
            else:
                numerator = (b_norm[i] * a[i] - b[i] * np.matmul(b[i].conj(), a[i])) * a[i, 0].conj()
                self.conv_weights[:, i] = numerator / denominator * self.__f[i].conj()
                # block matrix
                temp = -a[i, 1:] / a[i, 0]
                temp = temp[np.newaxis, :]
                block_matrix_1 = np.concatenate((temp, np.eye(self.mic_array.num_mic - 1)), axis=0)
                denominator = a[i, 1] * b[i, 0] - b[i, 1] * a[i, 0]
                numerator = -(a[i, 2:] * b[i, 0] - b[i, 2:] * a[i, 0])
                temp = numerator / denominator
                temp = temp[np.newaxis, :]
                block_matrix_2 = np.concatenate((temp, np.eye(self.mic_array.num_mic - 2)), axis=0)
                self.block_matrix[i, :, :] = np.matmul(block_matrix_2.T, block_matrix_1.T)

    # Several steps to check the validity of the TF and ensure that we only adopt accurate TFs
    def __check_tf_validity(self, h, incident_angle=None):
        # Step 1: check magnitude (magnitude is low when there is no speech or SNR is low)
        mag = np.mean(np.real(h * h.conj()))
        if mag < 0.5:
            return False
        # Step 2: check the time domain representation of the RIR and make sure there are only 1 peak for each channel
        interp_factor = 8
        nfft = ((h.shape[0] - 1) << 1) * interp_factor
        ht = np.fft.irfft(h[:, 1:].T, n=nfft, axis=1)
        ht = np.fft.fftshift(ht, axes=1)
        ht = ht[:, (nfft >> 1) - self.max_delay * interp_factor:(nfft >> 1) + self.max_delay * interp_factor + 1]
        peak_indices, peak_values = detect_peak(ht, max_number_peaks=2, sub_peak_thres=0.5)
        num_peaks = []
        for t in peak_indices:
            num_peaks.append(t.shape[0])
        if max(num_peaks) > 1:
            return False
        # Step 3: check the incident angle
        if incident_angle is not None:
            tau = np.array([((index[0] - (ht.shape[1] >> 1)) / self.sample_rate / interp_factor)
                            for index in peak_indices])
            incident_dir = np.matmul(self.pinv_factor, -tau) * self.sound_speed
            angle = incident_dir_to_angle(incident_dir)
            # print(angle.azimuth, angle.elevation)
            if abs(angle.azimuth - incident_angle.azimuth) > 5 or abs(angle.elevation - incident_angle.elevation) > 5:
                return False
        return True

    def __weights_fir_truncate(self, weights):
        if self.filter_length <= 0 or self.filter_length >= self._fft_length:
            return weights
        else:
            coefficients = np.fft.irfft(weights, axis=1)
            coefficients = coefficients[:, :self.filter_length]
            return np.fft.rfft(coefficients, n=self._fft_length, axis=1)


# R-GSC beamformer
class RGscBeamformer(Beamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed, filter_length, bm_learning_rate=0.1,
                 sc_learning_rate=0.2):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed)
        self.__conv_weights = np.array([1 / self.mic_array.num_mic] * self.mic_array.num_mic)
        self.__filter_length = filter_length
        self.__bm_delay = 5
        self.__bm_weights = np.zeros((self.mic_array.num_mic, self.__filter_length))
        self.__bm_weights[:, self.__bm_delay] = 1
        self.__bm_weights_high = None
        self.__bm_weights_low = None
        self.__init_bm_weights_constraints()
        self.__sc_delay = 5
        self.__sc_weights = np.zeros((self.mic_array.num_mic, self.__filter_length))
        self.__sc_weights[:, self.__sc_delay] = 1
        self.__bm_learning_rate = bm_learning_rate
        self.__sc_learning_rate = sc_learning_rate
        assert self.__filter_length >= 16
        self.__data_buffer = np.zeros([mic_array.num_mic, self.__filter_length - 1])

    def process(self, data, incident_angle=None):
        buffer_length = self.__data_buffer.shape[1]
        # Conventional path
        if incident_angle is not None:
            self.update_incident_angle(incident_angle)
        data = np.concatenate((self.__data_buffer, data), axis=1)
        self.__data_buffer = data[:, -buffer_length:]
        data_steered = self._steer(data)
        y_conv = np.matmul(self.__conv_weights, data_steered)

        # Sidelobe Cancelling path
        y = np.zeros((self.mic_array.num_mic, y_conv.shape[0]))
        z = np.zeros_like(y_conv)
        hd = np.zeros_like(y)
        wy = np.zeros_like(y)
        for i in range(buffer_length, y_conv.shape[0]):
            d = y_conv[i - self.__filter_length + 1:i + 1]
            d_delayed = y_conv[i - self.__bm_delay - self.__sc_delay]
            x_delayed = data_steered[:, i - self.__bm_delay]
            y_temp, hd_temp = self.__bm_process(d, x_delayed)
            y[:, i] = y_temp
            hd[:, i] = hd_temp
            z_temp, wy_temp = self.__sc_process(y[:, i - self.__filter_length + 1:i + 1], d_delayed)
            z[i] = z_temp
            wy[:, i] = wy_temp
        # save_wave_data("y.wav", (y * 32768).astype(np.int16))
        # save_wave_data("yfbf.wav", (y_conv * 32768).astype(np.int16))
        # save_wave_data("wy.wav", (wy * 32768).astype(np.int16))
        # save_wave_data("hd.wav", (hd * 32768).astype(np.int16))
        return z

    # block matrix process
    # d: N * 1 vector
    # x: m * 1 vector
    def __bm_process(self, d, x, update=True):
        # y = x - h.T * d
        # h <- h + alpha * y * d / p
        hd = np.matmul(self.__bm_weights, d)
        y = x - hd
        if update:
            p = np.square(np.linalg.norm(d))
            new_weights = self.__bm_weights + self.__bm_learning_rate * cartesian_product(y, d) / p
            new_weights = np.where(new_weights < self.__bm_weights_high, new_weights, self.__bm_weights_high)
            self.__bm_weights = np.where(new_weights > self.__bm_weights_low, new_weights, self.__bm_weights_low)
        return y, hd

    # sidelobe canceller process
    # y: m * L vector
    # d: scalar
    def __sc_process(self, y, d, threshold=10, update=True):
        # z = d - w.T * y
        # w <- w + beta * z * y / p
        wy = np.sum(self.__sc_weights * y, axis=1)
        z = d - np.sum(wy)
        if update:
            p = np.linalg.norm(y, axis=1)
            p = np.sum(np.square(p))
            new_weights = self.__sc_weights + self.__sc_learning_rate * z * y / p
            omega = np.linalg.norm(new_weights, axis=1)
            omega = np.sum(np.square(omega))
            # print(omega)
            if omega > threshold:
                new_weights = new_weights * np.sqrt(threshold / omega)
            self.__sc_weights = new_weights
        return z, wy

    def __init_bm_weights_constraints(self, threshold=15):
        if isinstance(self.mic_array, LinearMicrophoneArray):
            if self.mic_array.axis == 'x':
                angle = Angle3D(90 + threshold, 0)
            elif self.mic_array.axis == 'y':
                angle = Angle3D(threshold, 0)
            else:
                angle = Angle3D(0, threshold)
        else:
            if self.mic_array.normal_axis == 'x':
                angle = Angle3D(0, threshold)
            elif self.mic_array.normal_axis == 'y':
                angle = Angle3D(0, threshold)
            else:
                angle = Angle3D(threshold, 0)

        # Calculate steer vectors for the threshold incident angle
        tau = self._get_tau_from_angle(angle)
        freq_dim = (self.__filter_length >> 1) + 1
        freq_bins = np.linspace(0, self.sample_rate / 2, freq_dim)
        h = np.zeros((self.mic_array.num_mic, freq_dim), dtype=np.complex128)
        for i in range(h.shape[0]):
            h[i] = np.exp(-1j * 2 * np.pi * tau[i] * freq_bins)
        # Calculate the transfer function between the fixed beamformer output and input channels
        h_sum = np.sum(h.T * self.__conv_weights, axis=1)
        h_left = h[0] / h_sum
        h_right = h[-1] / h_sum
        weights_left = np.fft.irfft(h_left)
        weights_right = np.fft.irfft(h_right)
        weights_left = np.concatenate((weights_left[-self.__bm_delay:], weights_left[:-self.__bm_delay]))
        weights_right = np.concatenate((weights_right[-self.__bm_delay:], weights_right[:-self.__bm_delay]))
        weights_min = np.where(weights_left < weights_right, weights_left, weights_right)
        weights_max = np.where(weights_left > weights_right, weights_left, weights_right)
        envelop_max = get_envelop(weights_max) + 0.1
        envelop_min = -get_envelop(-weights_min) - 0.1
        self.__bm_weights_high = np.tile(envelop_max, [self.mic_array.num_mic, 1])
        self.__bm_weights_low = np.tile(envelop_min, [self.mic_array.num_mic, 1])


class TrackingMode(Enum):
    fixed_direction = 0
    finite_tracking = 1
    free_tracking = 2


class FreqRGscBeamformer(AbstractFreqBeamformer):
    def __init__(self, mic_array, incident_angle, sample_rate, sound_speed,
                 fft_length=64, operating_freq=0, bm_learning_rate=0.1, sc_learning_rate=0.2,
                 vad_doa_process=False, vad_doa_processor_config=None, tracking_mode=TrackingMode.fixed_direction,
                 pre_filter=False, channel_gain_estimation=False, debug=False):
        super().__init__(mic_array, incident_angle, sample_rate, sound_speed, fft_length, operating_freq,
                         vad_doa_process=vad_doa_process, vad_doa_processor_config=vad_doa_processor_config,
                         debug=debug)
        self.__interval = self._time_freq_converter.interval
        self.__overlap = self._fft_length - self.__interval
        self.__filter_delay = self.__interval >> 1
        self.__conv_weights = None
        self.__block_matrix = None
        self.__pre_emphasis_factor = 0.9
        self.__de_emphasis_factor = 0.9
        # MVDR
        self.__mvdr_weights = None
        self.__p = None
        self.__mvdr_learning_rate = 0.1
        self.__mvdr = True
        self.__phi_n = None
        self.__phi_s = None
        self.__phi_x = None
        ratio = 1 if self._fft_length < 128 else self._fft_length // 128
        self.__mvdr_p_est = IterativeLinearSmoother(0.95 ** ratio)
        # Beam width in degrees
        self.__beam_width = 30
        self.__threshold_level = 4
        self.__nlp = True
        self.__nlp_thres = 0.5
        self.__comfort_noise = True
        np.random.seed(17253)
        self.__snr_based_mask = False
        self.__snr_based_mask_factor = 0.1
        # RGSC weights
        self.__bm_weights_t = np.zeros((self.mic_array.num_mic, self._fft_length))
        self.__bm_weights_high = None
        self.__bm_weights_low = None
        self.__sc_weights_t = np.zeros((self.mic_array.num_mic, self._fft_length))
        self.__init_bm_sc_weights()
        self.__bm_learning_rate = bm_learning_rate
        self.__sc_learning_rate = sc_learning_rate
        self.__cross_fade_factor = np.linspace(1, 0, self.__interval, endpoint=False)
        ratio = 1 if self._fft_length < 128 else self._fft_length // 128
        self.__bm_p_est = IterativeLinearSmoother(0.95 ** ratio)
        self.__sc_p_est = IterativeLinearSmoother(0.95 ** ratio)
        self.__incident_angle = incident_angle
        self.__reset_incident_angle(incident_angle)
        # Channel gain estimator
        self.__channel_gain_estimation = channel_gain_estimation
        self.__noise_est_inc_factor = 1.01
        self.__noise_est_update_factor = 0.9
        self.__channel_gain_update_cnt = 0
        self.__frames_per_channel_gain_update = 50
        self.__noise_est = None
        self.__channel_gain = IterativeLinearSmoother(0.995 ** ratio, np.full(self.mic_array.num_mic, 1.0))
        # SIR mask estimator
        self.__sir_mask_thres_high = None
        self.__sir_mask_thres_low = None
        self.__mel_mask_thres_high = None
        self.__mel_mask_thres_low = None
        self.__mel_mask_thres_low_2 = None
        self.__mel_mask_thres_high_2 = None
        # MEL filter banks
        self.__mel_filter_bank, self.__mel_filter_bank_trans = get_mel_filter_bank_pair(
            self.sample_rate, self._fft_length, 128)
        self.__init_sir_mask_thres()
        self.__x_power_smoothed = IterativeLinearSmoother(0.92 ** ratio)
        self.__d_power_smoothed = IterativeLinearSmoother(0.92 ** ratio)
        self.__x_null_power_smoothed = IterativeLinearSmoother(0.92 ** ratio)
        self.__mask_smooth_factor_low = 0.3
        self.__mask_smooth_factor_high = 0.95 ** ratio
        self.__mask_mean = IterativeLinearSmoother(self.__mask_smooth_factor_high)
        self.__mask_smooth_factor = (1 - np.square(np.linspace(0, 1, self.freq_dim))) *\
                                    (self.__mask_smooth_factor_high - self.__mask_smooth_factor_low) +\
                                    self.__mask_smooth_factor_low
        self.__mask_smoothed = IterativeLinearSmoother(self.__mask_smooth_factor)
        self.__speech_mask_smoothed = IterativeLinearSmoother(0 ** ratio)
        self.__speech_mask_estimator = McraNoiseEstimator(McraNoiseEstimatorConfig())
        self.__mask_min = 0.01
        self.__mask_max = 0.99
        self.__mask_limit = 0.99
        # Pre filter
        self.__pre_filter = None
        if pre_filter:
            self.__pre_filter = MultiChanPreFilter(self.mic_array.num_mic, McraNoiseEstimatorConfig())
        # TF estimator
        self.__tracking_mode = tracking_mode
        self.__doa_based_mask = False
        self.__tf_updated = False
        self.__lms_doa_est = True
        self.__doa_direct_set_tau = False
        self.__doa_interval_factor = 1
        self.__doa_buffer = np.zeros((self.mic_array.num_mic, self.__interval * self.__doa_interval_factor))
        if self.__lms_doa_est:
            self.__doa_estimator = FreqLmsDoaEstimator(self.mic_array, self.sample_rate, sound_speed,
                                                       find_second_peak=True, learning_rate=0.1,
                                                       filter_length=self.__interval * self.__doa_interval_factor,
                                                       circular_conv=True, normalized_gradient=True,
                                                       interp_factor=8, second_peak_thres=0.7)
        else:
            self.__doa_estimator = GccPhatDoaEstimator(self.mic_array, self.sample_rate, sound_speed,
                                                       find_second_peak=True, interp_factor=8, second_peak_thres=0.7)
        tf_update_time = 0.5
        self.__tf_update_frames = int(tf_update_time * self.sample_rate / self._time_freq_converter.interval)
        self.__last_tf_update = 0
        self.__tau_smoothed = IterativeLinearSmoother(0.9)
        self.__incident_angle_smoothed.azimuth = incident_angle.azimuth
        self.__incident_angle_smoothed.elevation = incident_angle.elevation

    def __init_bm_sc_weights(self):
        self.__bm_weights_t.fill(0)
        self.__bm_weights_t[:, -self.__filter_delay] = 1
        self.__bm_weights = np.fft.rfft(self.__bm_weights_t, axis=-1)
        self.__sc_weights_t.fill(0)
        self.__sc_weights_t[:, -self.__filter_delay] = 1 / self.mic_array.num_mic
        self.__sc_weights = np.fft.rfft(self.__sc_weights_t, axis=-1)

    def __tf_estimator_process(self, x_frame, frame_index=0):
        self.__doa_buffer[:, :self.__doa_buffer.shape[1] - self.__interval] = self.__doa_buffer[:, self.__interval:]
        self.__doa_buffer[:, self.__doa_buffer.shape[1] - self.__interval:] = x_frame
        if (frame_index & (self.__doa_interval_factor - 1)) == self.__doa_interval_factor - 1:
            if self.__lms_doa_est:
                angle, taus, h = self.__doa_estimator.process(self.__doa_buffer)
                h = np.fft.rfft(h, n=self._fft_length, axis=1)
                h_temp = h[1:]
                mag = np.mean(np.real(h_temp * h_temp.conj()))
            else:
                angle, _, taus = self.__doa_estimator.process(self.__doa_buffer)
                mag = 1
        else:
            return
        if self._debug:
            print("time: %f mag: %f" % (frame_index * self.__interval / self.sample_rate, mag))
        # Step 1: check magnitude (magnitude is low when there is no speech or SNR is low)
        # Step 2: check the time domain representation of the RIR and make sure there are only 1 peak for each channel
        if self.__tracking_mode == TrackingMode.finite_tracking and mag > 0.4 and angle[0] == angle[1]:
            if self._debug:
                print(angle[0].azimuth, angle[1].elevation)
            if abs(angle[0].azimuth - self.__incident_angle.azimuth) < 10 and\
                    abs(angle[0].elevation - self.__incident_angle.elevation) < 10:
                self.__tau_smoothed.update(np.concatenate((np.array([0]), taus[0])))
                self.__incident_angle_smoothed.azimuth =\
                    self.__incident_angle_smoothed.azimuth * 0.5 + angle[0].azimuth * 0.5
                self.__incident_angle_smoothed.elevation =\
                    self.__incident_angle_smoothed.elevation * 0.5 + angle[0].elevation * 0.5
                # self._steer_vectors = h.T
                if frame_index - self.__last_tf_update >= self.__tf_update_frames:
                    if not self.__doa_direct_set_tau:
                        self.__tau_smoothed.reset(self._get_tau_from_angle(self.__incident_angle_smoothed))
                    self.__update_tau(self.__tau_smoothed.data)
                    self.__tf_updated = True
                    self.__last_tf_update = frame_index
                    if self._debug:
                        print("Time: %f TF updated!"
                              % (frame_index * self._time_freq_converter.interval / self.sample_rate))
                        print(self.__tau_smoothed.data)
            elif self.__tf_updated and frame_index - self.__last_tf_update >= self.__tf_update_frames:
                self.__incident_angle_smoothed.azimuth = self.__incident_angle.azimuth
                self.__incident_angle_smoothed.elevation = self.__incident_angle.elevation
                self.update_incident_angle(self.__incident_angle)
                self.__update_steer_vectors()
                self.__tf_updated = False
                self.__last_tf_update = frame_index
                if self._debug:
                    print("Time: %f TF reset!" %
                          (frame_index * self._time_freq_converter.interval / self.sample_rate))
                    print(self.__tau_smoothed.data)
        elif self.__tracking_mode == TrackingMode.free_tracking:
            smoothing = False
            i = 0
            for ang in angle:
                if abs(ang.azimuth - self.__incident_angle.azimuth) < 10 and\
                        abs(ang.elevation - self.__incident_angle.elevation) < 10:
                    self.__tau_smoothed.update(np.concatenate((np.array([0]), taus[i])))
                    self.__incident_angle_smoothed.azimuth =\
                        self.__incident_angle_smoothed.azimuth * 0.5 + ang.azimuth * 0.5
                    self.__incident_angle_smoothed.elevation =\
                        self.__incident_angle_smoothed.elevation * 0.5 + ang.elevation * 0.5
                    smoothing = True
                    if frame_index - self.__last_tf_update >= self.__tf_update_frames:
                        if not self.__doa_direct_set_tau:
                            self.__tau_smoothed.reset(self._get_tau_from_angle(self.__incident_angle_smoothed))
                        self.__update_tau(self.__tau_smoothed.data)
                        self.__tf_updated = True
                        self.__last_tf_update = frame_index
                        if self._debug:
                            print("Time: %f TF updated!"
                                  % (frame_index * self._time_freq_converter.interval / self.sample_rate))
                            print(self.__tau_smoothed.data)
                    break
                i += 1
            if not smoothing:
                self.__incident_angle.azimuth = angle[0].azimuth
                self.__incident_angle.elevation = angle[0].elevation
                self.__incident_angle_smoothed.azimuth = self.__incident_angle.azimuth
                self.__incident_angle_smoothed.elevation = self.__incident_angle.elevation
                if self.__doa_direct_set_tau:
                    self.__tau_smoothed.reset(np.concatenate((np.array([0]), taus[0])))
                else:
                    self.__tau_smoothed.reset(self._get_tau_from_angle(self.__incident_angle_smoothed))
                self.__reset_tau(self.__tau_smoothed.data)
                self.__tf_updated = True
                self.__last_tf_update = frame_index
                if self._debug:
                    print("Time: %f TF reset!"
                          % (frame_index * self._time_freq_converter.interval / self.sample_rate))
                    print(self.__tau_smoothed.data)
        if self.__tracking_mode != TrackingMode.free_tracking and self.__doa_based_mask:
            max_angle_diff = 0
            min_angle_diff = 360
            for ang in angle:
                azimuth_diff = abs(ang.azimuth - self.__incident_angle.azimuth)
                elevation_diff = abs(ang.elevation - self.__incident_angle.elevation)
                max_angle_diff = max(max_angle_diff, azimuth_diff)
                max_angle_diff = max(max_angle_diff, elevation_diff)
                min_angle_diff = min(min_angle_diff, max(azimuth_diff, elevation_diff))
            if min_angle_diff < 20:
                self.__mask_limit = min(self.__mask_max, self.__mask_limit + 0.2)
            elif max_angle_diff > 45:
                self.__mask_limit = 0.1
            elif max_angle_diff > 30:
                self.__mask_limit = max(0.1, self.__mask_limit - 0.2)
            elif max_angle_diff > 20:
                self.__mask_limit = max(0.1, self.__mask_limit - 0.1)

    def process(self, data, incident_angle=None):
        data = np.concatenate((np.zeros((self.mic_array.num_mic, self.__interval)), data), axis=1)
        if incident_angle is not None:
            self.__reset_incident_angle(incident_angle)
        if self.vad_doa_processor is not None:
            self.vad_doa_processor.doa_process(data)
        # data = pre_emphasis(data, self.__pre_emphasis_factor)
        input_length = data.shape[1]
        xf = MultiChanFrameData(self._subband_fft(data))
        # Padding ending zeros for frame splitting
        internal_length = (xf.num_frames - 1) * self.__interval + self._fft_length
        append_values = np.zeros((data.shape[0], internal_length - input_length), dtype=data.dtype)
        x = np.append(data, append_values, axis=-1)
        # Padding leading zeros for filter delay
        padding_length = np.maximum(self.__overlap + self.__filter_delay, 2 * self.__filter_delay)
        internal_length = x.shape[1] + padding_length
        x_padding = np.zeros((x.shape[0], padding_length), dtype=data.dtype)
        x = np.concatenate((x_padding, x), axis=-1)
        d = np.zeros(internal_length, dtype=data.dtype)
        d_with_mask = np.zeros(self._fft_length + self.__overlap, dtype=data.dtype)
        x_with_mask = np.zeros(self._fft_length + self.__overlap + self.__filter_delay, dtype=data.dtype)
        z = np.zeros(internal_length, dtype=data.dtype)
        y = np.zeros((self.mic_array.num_mic, internal_length), dtype=data.dtype)
        wy = np.zeros((self.mic_array.num_mic, internal_length), dtype=data.dtype)
        hd = np.zeros((self.mic_array.num_mic, internal_length), dtype=data.dtype)
        ratio_all = np.zeros((xf.num_frames, xf.feature_dim))
        mask_all = np.zeros((xf.num_frames, xf.feature_dim))
        channel_gain_all = np.zeros((xf.num_frames, self.mic_array.num_mic))
        speech_mask_all = np.zeros((xf.num_frames, xf.feature_dim))
        last_mask = np.ones(xf.feature_dim)
        self.__mvdr_weights = np.copy(self.__conv_weights)
        self.__last_tf_update = 0
        # Loop through frames
        for frame_index in range(xf.num_frames):
            frame_start_index = frame_index * self.__interval + padding_length
            # Conventional path
            xf_frame = xf.get_frame_data(frame_index)
            if (self.__tracking_mode != TrackingMode.fixed_direction or self.__doa_based_mask)\
                    and frame_index != xf.num_frames - 1:
                x_frame = data[:, frame_index * self.__interval + self.__overlap:
                                  frame_index * self.__interval + self._fft_length]
                self.__tf_estimator_process(x_frame, frame_index=frame_index)
            if self.__pre_filter is not None:
                xf_frame = self.__pre_filter.process(xf_frame)
            if self.__channel_gain_estimation:
                self.__channel_gain_estimator_process(np.real(xf_frame * xf_frame.conj()))
                channel_gain_all[frame_index] = self.__channel_gain.data

            # yf = w.H * xf
            if self.__mvdr:
                df_frame = np.sum(self.__mvdr_weights.conj() * xf_frame, axis=0)
                # power estimation
                p = np.sum(np.real(xf_frame * xf_frame.conj()), axis=0)
                self.__mvdr_p_est.update(p)
            else:
                df_frame = np.sum(self.__conv_weights.conj() * xf_frame, axis=0)
            # Null path
            # xf_frame_null = np.zeros((self.__block_matrix.shape[1], xf.feature_dim), dtype=np.complex64)
            # for freq_index in range(xf.feature_dim):
            #     xf_frame_null[:, freq_index] = np.matmul(self.__block_matrix[freq_index, :, :],
            #                                              xf_frame[:, freq_index])
            # Interference mask
            conv_d_frame = np.sum(self.__conv_weights.conj() * xf_frame, axis=0)
            d_power = np.real(conv_d_frame * conv_d_frame.conj())
            x_power = np.square(np.sum(np.abs(self.__conv_weights) * np.abs(xf_frame), axis=0))
            # x_null_power = np.real(xf_frame_null * xf_frame_null.conj())
            # x_null_power = np.square(np.mean(np.sqrt(x_null_power), axis=0))
            self.__update_speech_mask(x_power)
            self.__x_power_smoothed.update(x_power)
            self.__d_power_smoothed.update(d_power)
            # self.__x_null_power_smoothed.update(x_null_power)
            speech_mask_all[frame_index] = self.__speech_mask_smoothed.data
            mask, ratio = self.__get_sir_mask(method=1)
            # mask *= speech_mask
            ratio_all[frame_index] = ratio
            mask_all[frame_index] = mask

            if self.__mvdr:
                # MVDR update
                # w <- P * (w - u * xf * yf.H) + F
                update = self.__mvdr_weights - (1 - mask) * (1 - mask) * self.__mvdr_learning_rate * xf_frame * df_frame.conj() / \
                         np.maximum(self.__mvdr_p_est.data, 1e-10)
                for freq_index in range(update.shape[1]):
                    update[:, freq_index] = np.matmul(self.__p[freq_index, :, :], update[:, freq_index])
                self.__mvdr_weights = update + self.__conv_weights

                # PHI update
                # if self.__phi_n is None:
                #     self.__phi_n = np.zeros((self.freq_dim, self.mic_array.num_mic, self.mic_array.num_mic,), dtype=np.complex128)
                #     for i in range(self.freq_dim):
                #         self.__phi_n[i] = cartesian_product(xf_frame[:, i], xf_frame[:, i].conj())
                #     self.__phi_s = np.copy(self.__phi_n)
                #     self.__phi_x = np.copy(self.__phi_n)
                # else:
                #     for i in range(self.freq_dim):
                #         n_factor = 0.99 + mask[i] * 0.01
                #         s_factor = 0.99 + (1 - mask[i]) * 0.01
                #         x_factor = 0.99
                #         phi = cartesian_product(xf_frame[:, i], xf_frame[:, i].conj())
                #         self.__phi_n[i] = n_factor * self.__phi_n[i] + (1 - n_factor) * phi
                #         self.__phi_s[i] = s_factor * self.__phi_s[i] + (1 - s_factor) * phi
                #         self.__phi_x[i] = x_factor * self.__phi_x[i] + (1 - x_factor) * phi
                #
                # if frame_index > 50:
                #     for freq_index in range(self.freq_dim):
                #         randm = np.random.random((self.mic_array.num_mic, self.mic_array.num_mic)) * 1e-10
                #         w, v = np.linalg.eig(self.__phi_s[freq_index] + randm)
                #         steer_vec = v[:, 0]
                #         temp = np.linalg.solve(self.__phi_n[freq_index] + randm, steer_vec)
                #         denom = np.matmul(steer_vec.conj(), temp)
                #         self.__mvdr_weights[:, freq_index] = temp / denom
                # self.__mvdr_weights = self._constrain_weight_norm(self.__mvdr_weights,
                #                                                   threshold=50, frame_index=frame_index)

                # if frame_index > 50:
                #     for freq_index in range(self.freq_dim):
                #         randm = np.random.random((self.mic_array.num_mic, self.mic_array.num_mic)) * 1e-10
                #         temp1 = np.linalg.solve(self.__phi_n[freq_index] + randm, self.__phi_s[freq_index])
                #         temp2 = np.linalg.solve(self.__phi_n[freq_index] + randm, self.__phi_x[freq_index])
                #         self.__mvdr_weights[:, freq_index] = temp1[:, 0] / np.trace(temp2)
                # self.__mvdr_weights = self._constrain_weight_norm(self.__mvdr_weights,
                #                                                   threshold=10, frame_index=frame_index)

            # Convert d to time domain and overlap add
            d_tmp = np.fft.irfft(df_frame, axis=-1) * self._time_freq_converter.reconstruct_window
            d[frame_start_index:frame_start_index + self._fft_length] += d_tmp
            # Roll over d with mask and x with mask
            d_with_mask[:-self.__interval] = d_with_mask[self.__interval:]
            d_with_mask[-self.__interval:] = 0
            x_with_mask[:-self.__interval] = x_with_mask[self.__interval:]
            x_with_mask[-self.__interval:] = 0
            # Get d with mask, overlap add
            if self.__nlp:
                df_with_mask_frame = df_frame * np.maximum(mask, min(self.__nlp_thres, self.__mask_limit))
                if self.__comfort_noise:
                    noise = self.__get_comfort_noise()
                    df_with_mask_frame += noise * np.sqrt(np.maximum((1 - mask * mask), 0))
            else:
                df_with_mask_frame = df_frame
            d_with_mask_tmp = np.fft.irfft(df_with_mask_frame, axis=-1) * self._time_freq_converter.reconstruct_window
            d_with_mask[-self._fft_length:] += d_with_mask_tmp
            x_with_mask_tmp = np.fft.irfft(xf_frame[-1] * (1 - mask), axis=-1) * self._time_freq_converter.reconstruct_window
            x_with_mask[-self._fft_length:] += x_with_mask_tmp
            # Adaptive BM
            # Take the overlap part of d and delayed x for adpative BM stage
            xbm_start_index = frame_start_index - self.__overlap - self.__filter_delay
            d_with_mask_frame = d_with_mask[:self._fft_length]
            x_frame = x[:, xbm_start_index + self.__overlap:xbm_start_index + self._fft_length]
            df_with_mask_frame = np.fft.rfft(d_with_mask_frame, axis=-1)
            y_frame, hd_frame = self.__bm_process(df_with_mask_frame, x_frame, update=(frame_index >= 3),
                                                  sir_mask=(last_mask + mask) * 0.5)
            y[:, xbm_start_index + self.__overlap:xbm_start_index + self._fft_length] = y_frame
            hd[:, xbm_start_index + self.__overlap:xbm_start_index + self._fft_length] = hd_frame
            # Build a complete frame of y and delayed d for adaptive SC stage
            y_frame = y[:, xbm_start_index:xbm_start_index + self._fft_length]
            x_with_mask_frame = x_with_mask[:self._fft_length]
            # x_frame_1 = x[-1:, xbm_start_index:xbm_start_index + self._fft_length]
            # y_frame = np.concatenate((y_frame[:-1], x_with_mask_frame[np.newaxis, :]), axis=0)
            yf_frame = np.fft.rfft(y_frame, axis=-1)
            dsc_start_index = frame_start_index - self.__overlap - 2 * self.__filter_delay
            d_frame = d[dsc_start_index + self.__overlap:dsc_start_index + self._fft_length]
            z_frame, wy_frame = self.__sc_process(yf_frame, d_frame, frame_index=frame_index, threshold=10,
                                                  update=(frame_index >= 3), sir_mask=last_mask)
            z[dsc_start_index + self.__overlap:dsc_start_index + self._fft_length] = z_frame
            wy[:, dsc_start_index + self.__overlap:dsc_start_index + self._fft_length] = wy_frame
            last_mask = mask
        ####################################
        if self._debug:
            import matplotlib.pyplot as plt
            # plt.plot(self.__sir_mask_thres_low)
            # plt.plot(self.__sir_mask_thres_high)
            # plt.show()
            plt.plot(self.__mel_mask_thres_low)
            plt.plot(self.__mel_mask_thres_high)
            plt.show()
            plt.figure(figsize=(15, 5))
            plt.imshow(ratio_all.T, aspect=655360 / (self._fft_length ** 2))
            plt.show()
            plt.figure(figsize=(15, 5))
            plt.imshow(mask_all.T, aspect=655360 / (self._fft_length ** 2))
            plt.show()
            plt.figure(figsize=(15, 5))
            plt.plot(channel_gain_all)
            plt.show()
            plt.figure(figsize=(15, 5))
            plt.imshow(np.clip(speech_mask_all, 0, 1).T, aspect=655360 / (self._fft_length ** 2))
            plt.show()
        ####################################
        beam = z[self.__filter_delay: self.__filter_delay + input_length - self.__interval]
        if self._debug:
            save_wave_data("z.wav", (beam * 32768).astype(np.int16))
            # if self.vad_doa_processor is not None and self.__pre_filter is None:
            #     beam = self.vad_doa_processor.filter_process(beam)
            save_wave_data("y.wav", (y * 32768).astype(np.int16))
            save_wave_data("wy.wav", (wy * 32768).astype(np.int16))
            save_wave_data("hd.wav", (hd * 32768).astype(np.int16))
            save_wave_data("d.wav", (d * 32768).astype(np.int16))
            # d_deemp = de_emphasis(d, self.__de_emphasis_factor)
            # save_wave_data("d_deemp.wav", (d_deemp * 32768).astype(np.int16))
        return beam

    def __update_speech_mask(self, x_power):
        self.__speech_mask_estimator.update(x_power)
        speech_mask = self.__speech_mask_estimator.get_filter()
        self.__speech_mask_smoothed.update(speech_mask)

    def __get_comfort_noise(self):
        if self.__speech_mask_estimator.get_noise_estimation() is None:
            noise_power = np.zeros(self.freq_dim)
        else:
            noise_power = self.__speech_mask_estimator.get_noise_estimation()
        random_phase = 1j * 2 * np.pi * np.random.rand(noise_power.shape[0])
        random_phase[0] = 0
        random_phase[-1] = 0
        comfort_noise = np.exp(random_phase) * np.sqrt(noise_power) * 0.5
        comfort_noise[0] = 0
        return comfort_noise

    def __get_sir_mask(self, method=1):
        # Smoothing power on the frequency domain
        if method == 1:
            if self.__snr_based_mask:
                snr = (np.log(self.__speech_mask_estimator.get_snr_estimation()) - 1) * 0.5
                snr = np.clip(snr, 0, 1)
                scale_factor = (1 - self.__snr_based_mask_factor) + self.__snr_based_mask_factor * snr
                x_power = self.__x_power_smoothed.data * scale_factor
                d_power = self.__d_power_smoothed.data
            else:
                d_power = self.__d_power_smoothed.data
                x_power = self.__x_power_smoothed.data
            mel_ratio = self.__get_mel_ratio(d_power, x_power)
            mel_mask = (mel_ratio - self.__mel_mask_thres_low) /\
                       (self.__mel_mask_thres_high - self.__mel_mask_thres_low)
        else:
            mel_ratio = self.__get_mel_ratio(self.__x_null_power_smoothed.data, self.__d_power_smoothed.data)
            mel_mask = (mel_ratio - self.__mel_mask_thres_high_2) / \
                       (self.__mel_mask_thres_low_2 - self.__mel_mask_thres_high_2)
        mask = np.matmul(self.__mel_filter_bank_trans, mel_mask)
        ratio = np.matmul(self.__mel_filter_bank_trans, mel_ratio)
        mask = np.clip(mask, self.__mask_min, self.__mask_max)
        # Special handling for low frequency under 1000Hz
        freq_bin_1k = int(np.ceil(1000 / self.sample_rate * self._fft_length))
        freq_bin_1k5 = int(np.ceil(1500 / self.sample_rate * self._fft_length))
        freq_bin_4k = int(np.ceil(4000 / self.sample_rate * self._fft_length))
        freq_bin_5k = int(np.ceil(5000 / self.sample_rate * self._fft_length))
        freq_bin_500 = int(np.ceil(500 / self.sample_rate * self._fft_length))
        # Weighted mean of mask between 100 and 4k Hz (use d_mag as weight)
        d_mag = np.sqrt(self.__d_power_smoothed.data)
        mask_mean = np.sum(mask[freq_bin_1k5:freq_bin_4k] * d_mag[freq_bin_1k5:freq_bin_4k]) /\
                    np.maximum(np.sum(d_mag[freq_bin_1k5:freq_bin_4k]), 1e-10)
        mask_update_rate = self.__mask_smooth_factor_high
        if self.__mask_mean.data is not None and mask_mean > self.__mask_mean.data:
            mask_update_rate = self.__mask_smooth_factor_low
        self.__mask_mean.set_factor(mask_update_rate)
        self.__mask_mean.update(mask_mean)
        mask_mean = self.__mask_mean.data
        # Adjusting the low freq part
        if mask_mean >= 0.8:
            mask[:freq_bin_1k] = np.maximum(mask[:freq_bin_1k], mask_mean * 0.8)

        if not self.__doa_based_mask:
            if mask_mean <= 0.1:
                self.__mask_limit = 0.1
            elif mask_mean > 0.4:
                self.__mask_limit = self.__mask_max
            else:
                self.__mask_limit = min(self.__mask_max, self.__mask_limit + 0.2)
        mask[:freq_bin_4k] = np.minimum(mask[:freq_bin_4k], self.__mask_limit)

        '''
        if mask_mean < 0.5:
            if mask_mean > 0.2:
                factor = np.linspace(mask_mean + 0.5, mask_mean * 1.5 + 0.25, freq_bin_4k)
                threshold = np.concatenate((np.full(freq_bin_1k5, 1),
                                            np.linspace(1, mask_mean, freq_bin_4k - freq_bin_1k5)))
                mask[:freq_bin_4k] = np.where(mask[:freq_bin_4k] > threshold,
                                              mask[:freq_bin_4k],
                                              mask[:freq_bin_4k] * factor)
            else:
                factor = np.concatenate((np.linspace(0.7, 0.4, freq_bin_1k5),
                                         np.full(freq_bin_4k - freq_bin_1k5, 0.4)))
                mask[:freq_bin_4k] *= factor
            mask[:freq_bin_4k] = np.maximum(mask[:freq_bin_4k], self.__mask_min)
        '''
        # Smoothing the high freq part
        '''
        high_mean = 0
        for i in range(5):
            start_freq_bin = freq_bin_5k + i * freq_bin_500
            end_freq_bin = min(start_freq_bin + freq_bin_1k, self.freq_dim)
            local_mean = np.mean(mask[start_freq_bin:end_freq_bin])
            if local_mean > high_mean:
                high_mean = local_mean
        if high_mean > 0.5:
            mask[freq_bin_5k:] = np.maximum(mask[freq_bin_5k:], high_mean)
        elif high_mean > 0.2:
            mask[freq_bin_5k:] = np.maximum(mask[freq_bin_5k:], (high_mean - 0.2) * high_mean / 0.3)
        '''
        # Special handling for non-speech frequency under 100Hz
        freq_bin_100 = int(np.floor(100 / self.sample_rate * self._fft_length))
        mask[:freq_bin_100] = self.__mask_min
        if self.__mask_smoothed.data is not None:
            mask_update_rate = np.where(mask > self.__mask_smoothed.data,
                                        self.__mask_smooth_factor_low, self.__mask_smooth_factor)
            self.__mask_smoothed.set_factor(mask_update_rate)
        self.__mask_smoothed.update(mask)
        mask = self.__mask_smoothed.data
        return mask, ratio

    def __init_sir_mask_thres(self):
        # steer_vec = self._get_steer_vectors_from_tau(self._get_tau_from_angle(Angle3D(130, 0)))
        # beam = np.sum(steer_vec, axis=1) / self.mic_array.num_mic
        # amp = np.abs(beam)
        # self.__sir_mask_thres_low = np.clip(amp * 0.99, 0.6, 0.99)
        # self.__sir_mask_thres_high = np.zeros_like(self.__sir_mask_thres_low)
        # length = self.freq_dim
        # self.__sir_mask_thres_low = np.cos(np.arccos(0.6) / (length * 0.3) * np.arange(length)) * 0.99
        # self.__sir_mask_thres_low = np.maximum(self.__sir_mask_thres_low, 0.6)
        # self.__sir_mask_thres_high = np.cos(np.arccos(0.8) / (length >> 1) * np.arange(length))
        # self.__sir_mask_thres_high = np.maximum(self.__sir_mask_thres_high, 0.8)
        # self.__mel_mask_thres_high = np.sqrt(np.matmul(self.__mel_filter_bank, self.__sir_mask_thres_high ** 2) /
        #                                      np.sum(self.__mel_filter_bank, axis=1))
        # self.__mel_mask_thres_low = np.sqrt(np.matmul(self.__mel_filter_bank, self.__sir_mask_thres_low ** 2) /
        #                                    np.sum(self.__mel_filter_bank, axis=1))
        accept_angle_low, accept_angle_high, reject_angle = self.__get_threshold_angles()
        # Calculate steer vectors for the threshold incident angle
        accept_tau_low = self._get_tau_from_angle(accept_angle_low)
        accept_tau_high = self._get_tau_from_angle(accept_angle_high)
        reject_tau = self._get_tau_from_angle(reject_angle)
        accept_h_low = self._get_steer_vectors_from_tau(accept_tau_low).T
        accept_h_high = self._get_steer_vectors_from_tau(accept_tau_high).T
        reject_h = self._get_steer_vectors_from_tau(reject_tau).T
        accept_d_low = np.sum(accept_h_low * self.__conv_weights.conj(), axis=0)
        accept_d_high = np.sum(accept_h_high * self.__conv_weights.conj(), axis=0)
        reject_d = np.sum(reject_h * self.__conv_weights.conj(), axis=0)
        accept_x_null_low = np.zeros((self.__block_matrix.shape[1], self.freq_dim), dtype=np.complex64)
        accept_x_null_high = np.zeros((self.__block_matrix.shape[1], self.freq_dim), dtype=np.complex64)
        for freq_index in range(self.freq_dim):
            accept_x_null_low[:, freq_index] = np.matmul(self.__block_matrix[freq_index, :, :],
                                                         accept_h_low[:, freq_index])
            accept_x_null_high[:, freq_index] = np.matmul(self.__block_matrix[freq_index, :, :],
                                                          accept_h_high[:, freq_index])
        accept_ratio_low = self.__get_mel_ratio(np.real(accept_d_low * accept_d_low.conj()),
                                                np.ones_like(accept_d_low, dtype=np.float64))
        accept_ratio_high = self.__get_mel_ratio(np.real(accept_d_high * accept_d_high.conj()),
                                                 np.ones_like(accept_d_high, dtype=np.float64))
        reject_ratio = self.__get_mel_ratio(np.real(reject_d * reject_d.conj()),
                                            np.ones_like(reject_d, dtype=np.float64))
        accept_ratio = np.minimum(accept_ratio_low, accept_ratio_high)
        if self.__threshold_level == 0:
            self.__mel_mask_thres_high = np.clip(0.4 * (accept_ratio - 0.002) + 0.6 * (reject_ratio - 0.01), 0.5, None)
            self.__mel_mask_thres_low = np.clip(0.0 * (accept_ratio - 0.002) + 1.0 * (reject_ratio - 0.01), 0.3, None)
        elif self.__threshold_level == 1:
            self.__mel_mask_thres_high = np.clip(0.5 * (accept_ratio - 0.002) + 0.5 * (reject_ratio - 0.01), 0.6, None)
            self.__mel_mask_thres_low = np.clip(0.1 * (accept_ratio - 0.002) + 0.9 * (reject_ratio - 0.01), 0.4, None)
        elif self.__threshold_level == 2:
            self.__mel_mask_thres_high = np.clip(0.6 * (accept_ratio - 0.001) + 0.4 * (reject_ratio - 0.0075), 0.7, None)
            self.__mel_mask_thres_low = np.clip(0.2 * (accept_ratio - 0.001) + 0.8 * (reject_ratio - 0.0075), 0.5, None)
        elif self.__threshold_level == 3:
            self.__mel_mask_thres_high = np.clip(0.7 * (accept_ratio - 0.001) + 0.3 * (reject_ratio - 0.0075), 0.72, None)
            self.__mel_mask_thres_low = np.clip(0.3 * (accept_ratio - 0.001) + 0.7 * (reject_ratio - 0.0075), 0.52, None)
        elif self.__threshold_level == 4:
            self.__mel_mask_thres_high = np.clip(0.8 * accept_ratio + 0.2 * (reject_ratio - 0.005), 0.75, None)
            self.__mel_mask_thres_low = np.clip(0.4 * accept_ratio + 0.6 * (reject_ratio - 0.005), 0.55, None)
        elif self.__threshold_level == 5:
            self.__mel_mask_thres_high = np.clip(0.9 * accept_ratio + 0.1 * (reject_ratio - 0.005), 0.75, None)
            self.__mel_mask_thres_low = np.clip(0.5 * accept_ratio + 0.5 * (reject_ratio - 0.005), 0.55, None)
        else:
            self.__mel_mask_thres_high = np.clip(1.0 * accept_ratio + 0.0 * (reject_ratio - 0.005), 0.75, None)
            self.__mel_mask_thres_low = np.clip(0.6 * accept_ratio + 0.4 * (reject_ratio - 0.005), 0.55, None)
        self.__sir_mask_thres_high = np.matmul(self.__mel_filter_bank_trans, self.__mel_mask_thres_high)
        self.__sir_mask_thres_low = np.matmul(self.__mel_filter_bank_trans, self.__mel_mask_thres_low)
        self.__mel_mask_thres_low_2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3,
                                                0.32, 0.34, 0.36, 0.38, 0.4, 0.43, 0.46, 0.5, 0.54, 0.58,
                                                0.63, 0.71, 0.85, 1.02, 1.16, 1.24, 1.28, 1.32, 1.36, 1.38,
                                                1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])
        self.__mel_mask_thres_high_2 = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5,
                                                 0.52, 0.54, 0.56, 0.58, 0.6, 0.66, 0.72, 0.8, 0.88, 0.96,
                                                 1.08, 1.2, 1.38, 1.7, 2.1, 2.3, 2.35, 2.4, 2.45, 2.48,
                                                 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])

    # block matrix process
    # df: N * 1 vector, in frequency domain
    # x: m * n matrix, in time domain
    def __bm_process(self, df, x, update=True, sir_mask=None):
        # y = x - h.H * d
        # h <- h + alpha * y * d / p
        if sir_mask is None:
            sir_mask = 1
        # compute y using old weights
        y = np.zeros((x.shape[0], self._fft_length), dtype=x.dtype)
        hdf = self.__bm_weights.conj() * df
        hd = np.fft.irfft(hdf, axis=-1)
        y[:, self.__overlap:] = (x - hd[:, self.__overlap:]) * self.__cross_fade_factor
        # compute y using new weights and cross-fade
        self.__bm_weights = np.fft.rfft(self.__bm_weights_t, axis=-1)
        hdf = self.__bm_weights.conj() * df
        hd = np.fft.irfft(hdf, axis=-1)
        y[:, self.__overlap:] += (x - hd[:, self.__overlap:]) * (1 - self.__cross_fade_factor)
        p = np.real(df * df.conj())
        self.__bm_p_est.update(p)
        if update:
            yf = np.fft.rfft(y, axis=-1)
            if not self.__nlp:
                df *= sir_mask
            new_weights = self.__bm_weights +\
                          self.__bm_learning_rate * df * yf.conj() * sir_mask / np.maximum(self.__bm_p_est.data, 1e-10)
            w_tmp = np.fft.irfft(new_weights, axis=-1)
            w_tmp[:, :self.__overlap] = 0
            w_tmp = np.where(w_tmp < self.__bm_weights_high, w_tmp, self.__bm_weights_high)
            w_tmp = np.where(w_tmp > self.__bm_weights_low, w_tmp, self.__bm_weights_low)
            self.__bm_weights_t = w_tmp
        return y[:, self.__overlap:], hd[:, self.__overlap:]

    # sidelobe canceller process
    # y: m * N matrix, in frequency domain
    # d: n * 1 vector, in time domain
    def __sc_process(self, yf, d, threshold=10.0, update=True, frame_index=0, sir_mask=None):
        # z = d - w.H * y
        # w <- w + beta * z * y / p
        if sir_mask is None:
            sir_mask = 0
        # compute z using old weights
        z = np.zeros(self._fft_length, dtype=d.dtype)
        wyf = self.__sc_weights.conj() * yf
        wy = np.fft.irfft(wyf, axis=-1)
        wy_sum = np.sum(wy, axis=0)
        z[self.__overlap:] = (d - wy_sum[self.__overlap:]) * self.__cross_fade_factor
        # compute z using new weights
        self.__sc_weights = np.fft.rfft(self.__sc_weights_t, axis=-1)
        wyf = self.__sc_weights.conj() * yf
        wy = np.fft.irfft(wyf, axis=-1)
        wy_sum = np.sum(wy, axis=0)
        z[self.__overlap:] += (d - wy_sum[self.__overlap:]) * (1 - self.__cross_fade_factor)
        p = np.sum(np.real(yf * yf.conj()), axis=0)
        self.__sc_p_est.update(p)
        if update:
            zf = np.fft.rfft(z, axis=-1)
            zf *= (1 - sir_mask)
            new_weights = self.__sc_weights +\
                          self.__sc_learning_rate * yf * zf.conj() * (1 - sir_mask) / np.maximum(self.__sc_p_est.data, 1e-10)
            w_tmp = np.fft.irfft(new_weights, axis=-1)
            w_tmp[:, :self.__overlap] = 0
            omega = np.linalg.norm(w_tmp, axis=1)
            omega = np.sum(np.square(omega))
            if omega > threshold:
                w_tmp = w_tmp * np.sqrt(threshold / omega)
            self.__sc_weights_t = w_tmp
        return z[self.__overlap:], wy[:, self.__overlap:]

    def __channel_gain_estimator_process(self, xf_frame_power):
        # update noise power
        freq_bin_50 = int(np.ceil(50 / self.sample_rate * self._fft_length))
        '''
        pxx = np.mean(xf_frame_power, axis=0)
        if self.__noise_est is None:
            self.__noise_est = np.full(self.__mel_filter_bank.shape[0], np.mean(pxx[freq_bin_50:]))
        else:
            pxx_mel = np.matmul(self.__mel_filter_bank, pxx)
            self.__noise_est = np.where(pxx_mel < self.__noise_est,
                                        self.__noise_est_update_factor * self.__noise_est + \
                                        (1 - self.__noise_est_update_factor) * pxx_mel,
                                        self.__noise_est)
            self.__noise_est *= self.__noise_est_inc_factor
        noise_power = np.matmul(self.__mel_filter_bank_trans, self.__noise_est)
        '''
        if self.__speech_mask_estimator.get_noise_estimation() is None:
            noise_power = np.zeros(self.freq_dim)
        else:
            noise_power = self.__speech_mask_estimator.get_noise_estimation()
        noise_power = noise_power[freq_bin_50:]
        signal_power = xf_frame_power[:, freq_bin_50:] - noise_power
        signal_power = np.clip(signal_power, 0, None)
        total_signal_power = np.sum(signal_power, axis=1)
        # Only update channel gain when signal power is larger than noise power
        if np.min(total_signal_power) > np.sum(noise_power):
            channel_gain = np.sqrt(total_signal_power / total_signal_power[0])
            self.__channel_gain.update(channel_gain)
            if self.__channel_gain_update_cnt >= self.__frames_per_channel_gain_update:
                self.__update_steer_vectors()
                self.__channel_gain_update_cnt = 0
            if self._debug:
                print("channel gain: ", self.__channel_gain.data)
        self.__channel_gain_update_cnt += 1

    def __reset_incident_angle(self, incident_angle):
        self.__incident_angle = incident_angle
        self.__incident_angle_smoothed = Angle3D(incident_angle.azimuth, incident_angle.elevation)
        self.update_incident_angle(incident_angle)
        self.__init_conv_weights_and_block_matrix()
        self.__init_bm_weights_constraints()
        if self.__mvdr:
            self.__update_p()

    def __update_tau(self, tau):
        self._steer_vectors_uniform_gain = self._get_steer_vectors_from_tau(tau)
        self.__update_steer_vectors()

    def __reset_tau(self, tau):
        self._steer_vectors_uniform_gain = self._get_steer_vectors_from_tau(tau)
        self._steer_vectors = np.copy(self._steer_vectors_uniform_gain)
        self.__init_conv_weights_and_block_matrix()
        self.__init_bm_weights_constraints()
        self.__init_sir_mask_thres()
        self.__update_steer_vectors()
        self.__init_bm_sc_weights()
        if self.__mvdr:
            self.__mvdr_weights = np.copy(self.__conv_weights)

    def __update_steer_vectors(self):
        self._update_channel_gain(self.__channel_gain.data)
        self.__init_conv_weights_and_block_matrix()
        if self.__mvdr:
            self.__update_p()

    def __update_p(self):
        self.__p = np.zeros((self._steer_vectors.shape[0], self._steer_vectors.shape[1], self._steer_vectors.shape[1]),
                            dtype=np.complex64)
        norm = np.sum(self._steer_vectors.conj() * self._steer_vectors, axis=1)
        norm = np.tile(norm, [self.mic_array.num_mic, 1])
        # P = I - a * a.H / (|a|^2)
        for i in range(self.__p.shape[0]):
            a = self._steer_vectors[i, :]
            self.__p[i, :, :] = np.eye(self.__p.shape[1]) - cartesian_product(a, a.conj()) / norm[:, i]

    def __init_conv_weights_and_block_matrix(self):
        # a0 = np.tile(self._steer_vectors[:, 0], (self.mic_array.num_mic, 1)).T
        h = self._steer_vectors
        norm = np.sum(h.conj() * h, axis=1)
        norm = np.tile(norm, [self.mic_array.num_mic, 1])
        # w0 = h / (|h|^2)
        self.__conv_weights = h.T / norm
        self.__block_matrix = np.zeros(
            (self._steer_vectors.shape[0], self.mic_array.num_mic - 1, self.mic_array.num_mic),
            dtype=np.complex64)
        for i in range(self.__block_matrix.shape[0]):
            self.__block_matrix[i, :, :] = np.ones((self.mic_array.num_mic - 1, self.mic_array.num_mic),
                                                   dtype=np.complex64)
            h_sum = np.sum(h[i])
            for j in range(self.__block_matrix.shape[1]):
                self.__block_matrix[i, j, j] = -(h_sum - h[i, j]) / h[i, j]
            self.__block_matrix[i, :, :] /= self.mic_array.num_mic - 1

    def __init_bm_weights_constraints(self):
        angle_low, angle_high, _ = self.__get_threshold_angles()
        # Calculate steer vectors for the threshold incident angle
        tau_low = self._get_tau_from_angle(angle_low)
        tau_high = self._get_tau_from_angle(angle_high)
        h_low = self._get_steer_vectors_from_tau(tau_low).T
        h_high = self._get_steer_vectors_from_tau(tau_high).T
        # Calculate the transfer function between the fixed beamformer output and input channels
        h_sum_low = np.sum(h_low * self.__conv_weights.conj(), axis=0)
        h_sum_high = np.sum(h_high * self.__conv_weights.conj(), axis=0)
        h_low = h_low / h_sum_low
        h_high = h_high / h_sum_high
        weights_left = np.fft.irfft(h_low)
        weights_right = np.fft.irfft(h_high)
        weights_all = np.concatenate((weights_left, weights_right), axis=0)
        weights_all = np.flip(weights_all, axis=1)
        weights_all = np.concatenate((weights_all[:, self.__filter_delay:], weights_all[:, :self.__filter_delay]),
                                     axis=1)
        weights_min = np.min(weights_all, axis=0)
        weights_max = np.max(weights_all, axis=0)
        envelop_max = get_envelop(weights_max)
        envelop_min = -get_envelop(-weights_min)
        envelop_max = np.maximum(envelop_max * 1.2, envelop_max + 0.1)
        envelop_min = np.minimum(envelop_min * 1.2, envelop_min - 0.1)
        self.__bm_weights_high = np.tile(envelop_max, [self.mic_array.num_mic, 1])
        self.__bm_weights_low = np.tile(envelop_min, [self.mic_array.num_mic, 1])

    def __get_threshold_angles(self):
        if isinstance(self.mic_array, LinearMicrophoneArray):
            azimuth_angle = not self.mic_array.axis == 'z'
        elif isinstance(self.mic_array, CircularMicrophoneArray):
            azimuth_angle = self.mic_array.normal_axis == 'z'
        else:
            azimuth_angle = True
        if azimuth_angle:
            if self.__incident_angle.azimuth >= 0:
                max_azimuth = 180
                min_azimuth = 0
            else:
                max_azimuth = 0
                min_azimuth = -180
            accept_low_azimuth = np.maximum(self.__incident_angle.azimuth - self.__beam_width / 2, min_azimuth)
            accept_high_azimuth = np.minimum(self.__incident_angle.azimuth + self.__beam_width / 2, max_azimuth)
            if 0 <= self.__incident_angle.azimuth < 90 or -180 <= self.__incident_angle.azimuth < -90:
                reject_azimuth = np.minimum(self.__incident_angle.azimuth + self.__beam_width, max_azimuth)
            else:
                reject_azimuth = np.maximum(self.__incident_angle.azimuth - self.__beam_width, min_azimuth)
            accept_angle_low = Angle3D(accept_low_azimuth, 0)
            accept_angle_high = Angle3D(accept_high_azimuth, 0)
            reject_angle = Angle3D(reject_azimuth, 0)
        else:
            max_elevation = 90
            min_elevation = -90
            accept_low_elevation = np.maximum(self.__incident_angle.elevation - self.__beam_width / 2, min_elevation)
            accept_high_elevation = np.minimum(self.__incident_angle.elevation + self.__beam_width / 2, max_elevation)
            if self.__incident_angle.elevation > 0:
                reject_elevation = np.maximum(self.__incident_angle.elevation - self.__beam_width, min_elevation)
            else:
                reject_elevation = np.minimum(self.__incident_angle.elevation + self.__beam_width, max_elevation)
            accept_angle_low = Angle3D(0, accept_low_elevation)
            accept_angle_high = Angle3D(0, accept_high_elevation)
            reject_angle = Angle3D(0, reject_elevation)
        return accept_angle_low, accept_angle_high, reject_angle

    def __get_mel_ratio(self, x_power, y_power):
        return np.sqrt(np.matmul(self.__mel_filter_bank, x_power) / \
                       np.maximum(np.matmul(self.__mel_filter_bank, y_power), 1e-10))
