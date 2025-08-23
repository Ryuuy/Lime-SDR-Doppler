#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <lime/LimeSuite.h>     // Required to drive LimeSDR
#include <gnuplot_i.hpp>        // Required for Plotting
#include <fftw3.h>              // Required for DFT and FFT
#include <unistd.h>
#include <string>
#include <fstream>
#include <cstring>  // for memcpy
#include <sstream>
#include <sys/stat.h>   // 用于创建目录
#include <sys/types.h>  // 用于目录操作
#include <thread>   // 用于创建目录
#include <chrono>  // 用于目录操作
#include <cstdlib> // For system()

void EnsureFolderExists(const std::string& folderPath) {
    std::string command = "mkdir -p " + folderPath; // -p 确保递归创建目录
    system(command.c_str());
}


int SDRerror(const char *string, lms_device_t* device)
{
    printf("--------------------------------------------------\n"
           "SDR Error: %s\n"
           "--------------------------------------------------", string);
    if (device != NULL)
        LMS_Close(device);
    exit(-1);
}

int inputError(const char *string)
{
    printf("--------------------------------------------------\n"
           "Input Error: %s\n"
           "--------------------------------------------------", string);
}

void saveBufferedDataToFile(int frameNumber, const std::vector<int16_t>& buffer, const std::string& folderName) {
    // 创建文件名，并将其保存在指定文件夹中
    std::ostringstream fileName;
    fileName << folderName << "/sdr_data_frame_" << frameNumber << ".bin";  // 使用二进制格式存储
    
    std::ofstream outFile(fileName.str(), std::ios::out | std::ios::binary);
    if (outFile.is_open()) {
        // 将整个缓冲区批量写入文件
        outFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(int16_t));
        outFile.close();
        std::cout << "Frame " << frameNumber << " saved to " << fileName.str() << std::endl;
    } else {
        std::cerr << "Error opening file " << fileName.str() << std::endl;
    }
}

// 创建目录
void createDirectory(const std::string& folderName) {
    if (mkdir(folderName.c_str(), 0777) == -1) {
        std::cerr << "Error creating directory " << folderName << std::endl;
    } else {
        std::cout << "Directory " << folderName << " created" << std::endl;
    }
}


//**************************************************************************************
class Lime_SDR_mini_Rx
{
public:
    lms_device_t* device;
    lms_stream_t streamId[2]; // 两个RX Stream

    Lime_SDR_mini_Rx()
    {
        device = NULL;
    };

    ~Lime_SDR_mini_Rx()
    {
        Close_Stream();
        LMS_Close(device);
    };

    int Setup_SDR(float_type centerFreq, float_type sampleRate, unsigned gain)
    {
        // Open the first device
        lms_info_str_t list[8];
        int n;

        if ((n = LMS_GetDeviceList(list)) < 0)
            SDRerror("No devices found", device);

        if (LMS_Open(&device, list[0], NULL))
            SDRerror("Device could not be opened", device);

        if ((n = LMS_GetNumChannels(device, LMS_CH_RX)) < 0)
            error();


        // Enable RX channels
        for (int i = 0; i < 2; ++i) {
            if (LMS_EnableChannel(device, LMS_CH_RX, i, true) != 0)
                error();

            if (LMS_SetLOFrequency(device, LMS_CH_RX, i, centerFreq) != 0)
                error();
        }

        // Load configuration file
        const char* config_file = "./1895mimo.ini";
        if (LMS_LoadConfig(device, config_file) == 0) {
            printf("配置文件加载成功\n");
        } else {
            printf("配置文件加载失败\n");
        }
        // for (int i = 0; i < 2; ++i) {
        // if (LMS_Calibrate(device, LMS_CH_RX, i, sampleRate, 0)!= 0)
        //     error();
        // }
        // Setup and start streams
        Setup_Stream();
        Start_Stream();

        return 1;
    }
    void error() {
    std::cerr << "An error occurred." << std::endl;
    exit(EXIT_FAILURE); // 退出程序
    }
    int Setup_Stream()
    {
        for (int i = 0; i < 2; ++i) {
            streamId[i].channel = i;                 // Channel number
            streamId[i].fifoSize = 1024 * 1024;      // FIFO size in samples
            streamId[i].throughputVsLatency = 1.0;   // Optimize for max throughput
            streamId[i].isTx = false;               // RX channel
            streamId[i].dataFmt = lms_stream_t::LMS_FMT_I12; // 12-bit integers

            // Setup each stream
            if (LMS_SetupStream(device, &streamId[i]) != 0)
                SDRerror("Stream could not be set up", device);
        }
        return 1;
    }

    void Start_Stream()
    {
        // Start streaming for each stream
        for (int i = 0; i < 2; ++i) {
            if (LMS_StartStream(&streamId[i]) != 0)
                SDRerror("Stream could not be started", device);
        }
    }

    void Stop_Stream()
    {
        // Stop streaming for each stream
        for (int i = 0; i < 2; ++i) {
            if (LMS_StopStream(&streamId[i]) != 0)
                SDRerror("Stream could not be stopped", device);
        }
    }

    void Close_Stream()
    {
        // Destroy each stream
        for (int i = 0; i < 2; ++i) {
            if (LMS_DestroyStream(device, &streamId[i]) != 0)
                SDRerror("Stream could not be destroyed", device);
        }
    }
    void Change_RF_Settings(float_type centerFreq = 0, float_type sampleRate = 0, unsigned gain = 100)
        {
            if(centerFreq != 0)
            {
                //Set center frequency
                if (LMS_SetLOFrequency(device, LMS_CH_RX, 0, centerFreq) != 0)
                    SDRerror("Center Frequency could not be set", device);
            }
            // if(sampleRate != 0)
            // {
            //     //Set sample rate
            //     //This sets sampling rate for all channels
            //     if (LMS_SetSampleRate(device, sampleRate, 0) != 0)
            //         SDRerror("Sample rate could not be set", device);
            // }
            // if(gain != 100)
            // {
            //     if(LMS_SetGaindB(device, LMS_CH_RX, 0, gain)!=0)
            //         SDRerror("Gain could not be set", device);
            // }
        }

};

//**************************************************************************************
//#Class:       FFT_Settings
//**************************************************************************************
//#Description: This class manages the Fast Fourier Transformations. The FFT Length is
//              only changeable through this class
//**************************************************************************************
//#Members:     Public
//#Variables
//              <DFTlength>             :[---] <int> Fourier Transform Length
//              <in>                    :[---] <fftw_complex*>(double*) Input to FFT
//              <out>                   :[---] <fftw_complex*>(double*) Output from FFT
//#Functions
//              <FFT_Settings>          :Constructor
//              <~FFT_Settings>         :Destructor
//              <Execute_FFTW>          :Performs FFT with <in> and saves result to <out>
//              <Change_FFT_Settings>   :Changes FFT Length and adapts size of <in> and
//                                       <out>
//              Private
//#Variables
//              <p>                     :<fftw_plan> this structure contains FFT settings
//**************************************************************************************
//#Date         18.01.2022
//**************************************************************************************
//#Author       Fabian Török
//**************************************************************************************
class FFT_Settings
{
    private:
        fftw_plan p;

    public:
        int DFTlength;
        fftw_complex *in;
        fftw_complex *out;
        FFT_Settings(int DFTlengthIn)

        {
            DFTlength = DFTlengthIn;
            // allocate fftw_complex array of the size DFTlength * 2
            in = (fftw_complex*) fftw_malloc(DFTlength * sizeof(fftw_complex));
            out = (fftw_complex*) fftw_malloc(DFTlength * sizeof(fftw_complex));
            // declare FFTLength, input and output as well as Forward FFT
            p = fftw_plan_dft_1d(DFTlength, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        };

        void Execute_FFTW()
        {
            fftw_execute(p);
        };

        void Change_FFT_Settings(int DFTlengthIn)
        {
            // Free old memory
            fftw_free(in);
            fftw_free(out);
            fftw_destroy_plan(p);

            // Set new values
            DFTlength = DFTlengthIn;
            in = (fftw_complex*) fftw_malloc(DFTlength * sizeof(fftw_complex));
            out = (fftw_complex*) fftw_malloc(DFTlength * sizeof(fftw_complex));
            p = fftw_plan_dft_1d(DFTlength, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        };


        ~FFT_Settings()
        {
            // Free dynamically allocated memory
            fftw_destroy_plan(p);
            fftw_free(in);
            fftw_free(out);
        };
};

//**************************************************************************************
//#Class:       FFT_Settings
//**************************************************************************************
//#Description: Each object of this class manages an own Trace window. The classes FFT_
// Settings and Lime_SDR_mini_Rx are "helper" classes for this class. The user specifies
// the desired Spectrum Analyzer settings in the Constructor of this class. The class
// then takes care of setting the required RF Settings in the SDR and preparing the FFT
// Transformation.
//
// Using the method "Aqcuire_Sweep", the spectrum plot will be initiated.
//**************************************************************************************
//#Members:     Public
//#Functions
//              <Trace>                 :Constructor
//              <~Trace>                :Destructor
//              <Aqcuire_Sweep>         :Performs a sweep. If user defined span is
//                                       greater than the resolution bandwidth, multiple
//                                       frames of <FFTLength> are aquired after retuning
//                                       the SDR
//              <Plot_Spectrum>         :Plots content of XAxisBuffer and YAxisBuffer
//
//              Private
//#Variables
//              <SDRSetting>            :<SDRSettings> struct of SDR settings
//              <DispSetting>           :<DisplaySettings> struct of RF settings of
//                                       Spectrum Analyzer
//              <PlotSetting>           :<PlotSettings> struct of Plot specific settings
//              <StreamBuffer>          :<int16_t*> array for storing stream data
//              <XAxisbuffer>           :<double*> array for storing XAxis plot data
//              <YAxisbuffer>           :<double*> array for storing YAxis plot data
//              <YAxisbufferAvg>        :<double*> array for storing YAxis plot for
//                                       averaging
//              <GnuplotHandle>         :<Gnuplot> Gnuplot object
//              <FFT>                   :<FFT_Settings> FFT_Settings object
//              <SDR>                   :<Lime_SDR_mini_Rx> Lime_SDR_mini_Rx object
//              <AveragingCounter>      :<int> counter to track saved frames for
//                                       averaging
//              <initAveraging>         :<bool> flag indicating that first averaging
//                                       cycle has finished
//#Functions
//              <getFFTLength>          :returns current FFT length
//              <Change_RF_Settings>    :translates user requirements and changes SDR
//                                       RF settings as well
//
//**************************************************************************************
//#Date         18.01.2022
//**************************************************************************************
//#Author       Fabian Török
//**************************************************************************************
class Trace
{
    public:
        Trace(float_type centerFreqIn, float_type dispSpanIn,  float_type resBWIn, double gainIn, double refAmpIn, int FFTIn, int averaging, unsigned TraceNum):FFT(8192)
        {
            // Apply Display settings
            Display.centerFreq = centerFreqIn;
            Display.span = dispSpanIn;
            Display.resBW = resBWIn;// maybe the baseband signal frequency
            Display.refAmp = refAmpIn;
            Display.traceStatus = false;
            // Flag marking that first batch of averaging values have been recorded

            if(averaging > 0)
            {
                Display.averagingCntr = 0;
                Display.nAveraging = averaging;
                initAveraging = false;
                AveragingCounter = 1;
            }
            else
            {
                Display.averagingCntr = -1;

            }

            // Detector settings
            Detector.clearWrite = true;
            Detector.maxHold = false;
            Detector.minHold = false;

            // Measurement Settings
            for(int x = 0; x<3; x++)
                Meas.marker[x] = 0;
            Meas.chMeasurementBW = 0;

            // Specify SDR sample rate according to Nyquist Theorem
            SDRSetting.sampleRate = resBWIn;
            // Calculate time for SDR to sample a frame
            SDRSetting.frameTime = 1/SDRSetting.sampleRate*FFTIn;
            // Set initial Center Frequency to beginning of span. Note: the term Center Frequency is misleading here as the frequency set as "Center Frequency"
            // in the SDR will from where the downmixed Basband signal starts. ->Start Frequency is a better name here
            SDRSetting.centerFreq = centerFreqIn;//  starting freq actually
            SDRSetting.gain = gainIn;
            FFT.Change_FFT_Settings(FFTIn);
            // Length of the dynaminc arrays equals FFTIn/2*(Display.span/Display.resBW --> FFTIn/2 is sufficient here as the other half of the FFT will be redundant
            // information. See Nyquist Theorem.

            // Stream will fit the data as follows I,Q,I,Q,I,Q... --> double the size of DFTlength
            StreamBuffer = new int16_t[FFT.DFTlength * 2];
            // Setup SDR
            SDR.Setup_SDR(SDRSetting.centerFreq, SDRSetting.sampleRate, SDRSetting.gain);
        };
        //**************************************************************************************
        //**************************************************************************************
        ~Trace()
        {
            //free dynamically allocated arrays
            //free(YAxisBuffer);
            //free(XAxisBuffer);
            //free(StreamBuffer);
            //free(YAxisBufferAvg);

            delete [] YAxisBuffer;
            delete [] YAxisBufferAvg;
            delete [] XAxisBuffer;
            delete [] StreamBuffer;
            delete [] Meas.chMeasurementLabelID;
            delete [] Meas.labelID;

            // Delete double pointer array
            for( int i = 0 ; i < 3 ; i++ ) {
                delete [] Meas.markerLabelID[i];
            }
            delete [] Meas.markerLabelID;
        };
        
        void Acquire_Sweep_Buffered_MultiChannel(int frameCounter, const std::vector<std::string>& folderNames,

            int FFTLength, std::vector<int16_t*>& streamBuffers,
            
            int bufferSize, int repeatNumber, int numChannels)
            
            {
            
            for (int z = 0; z < (int)ceil(Display.span / Display.resBW); z++) {
            
            // 设置频率
            
            double currentFreq = Display.centerFreq + 10e6 * z;//SDRSetting.sampleRate
            
            // 每个频率重复采集 repeatNumber 次
            
            for (int i = 0; i < repeatNumber; i++) {
            
            for (int ch = 0; ch < numChannels; ++ch) {
            
            // 从 LimeSDR 接收样本
            
            int samplesRead = LMS_RecvStream(&SDR.streamId[ch], streamBuffers[ch], FFTLength, NULL, 3000);
            
            
            if (samplesRead > 0) {
            
            // 创建文件路径
            
            std::ostringstream fileName;
            
            fileName << folderNames[ch]
            
            << "/center" << static_cast<int>(currentFreq / 1e6) // 将频率转换为 MHz
            
            << "frame" << frameCounter
            
            << ".bin";
            
            
            // 打开文件并写入数据
            
            std::ofstream outFile(fileName.str(), std::ios::binary);
            
            if (!outFile.is_open()) {
            
            std::cerr << "Failed to open file: " << fileName.str() << std::endl;
            
            continue; // 如果文件无法打开，跳过
            
            }
            
            
            outFile.write(reinterpret_cast<const char*>(streamBuffers[ch]), samplesRead * 2 * sizeof(int16_t));
            
            outFile.close();
            
            } else {
            
            std::cerr << "Failed to receive samples for channel " << ch
            
            << ", freq " << z << ", repeat " << i << std::endl;
            
            }
            
            }
            
            
            // 更新 frameCounter
            
            frameCounter++;
            
            }
            
            
            std::cout << "Next saved frame " << frameCounter << ", freq " << z << std::endl;
            
            }
            
            
            } 



    private:
        // Diplay settings struct
        struct DispSettings
        {
            float_type centerFreq;
            float_type resBW;
            float_type span;
            double refAmp;
            int nAveraging;
            int averagingCntr;
            bool traceStatus;
        };
        // Detector struct
        struct DetectorSettings
        {
            bool clearWrite;
            bool maxHold;
            bool minHold;
        };
        // SDR settings struct
        struct SDRSettings
        {
            unsigned gain;
            float_type centerFreq;
            float_type sampleRate;
            float_type frameTime;
        };
        // Measurement settings
        struct MeasSettings
        {
            float marker[3] = {0};
            float chMeasurementBW = 0;
            float markerLabelPos[3][2] = {{0},{0}};
            int **markerLabelID = NULL;
            float chMeasurementLabelPos[2] = {0};
            int *chMeasurementLabelID = NULL;
            int *labelID = NULL;
            int labelCounter = 0;
        };
        // Plot settings struct
        struct PlotSettings
        {
            float_type refAmp;
            FILE* plotFile;
            char* fileName;
        };

        // declare private variables
        SDRSettings SDRSetting;
        DispSettings Display;
        DetectorSettings Detector;
        MeasSettings Meas;
        int16_t *StreamBuffer;
        double *YAxisBuffer;
        double *YAxisBufferAvg;
        double *XAxisBuffer;
        Gnuplot GnuplotHandle = Gnuplot("lines");
        FFT_Settings FFT = FFT_Settings(8192);
        Lime_SDR_mini_Rx SDR = Lime_SDR_mini_Rx();
        int AveragingCounter;
        bool initAveraging;

        // Returns FFT Length
        int getFFTLength()
        {
            return FFT.DFTlength;
        };

        // This function manages all label IDs used in gnuplot
        void assignLabelID(int* assignedLabels, int labelLength)
        {
            int pos = 0;
            bool flag;
            for(int x = 0; x<Meas.labelCounter; x++)
            {
               if(Meas.labelID[x] == 0)
               {
                  for(int y = 0; y<labelLength-1; y++)
                  {
                     if(Meas.labelID[x+y] == 0)
                         flag = true;
                     else
                         flag = false;
                  }
                  if(flag == true)
                  {
                      pos = x;
                      break;
                  }
               }
               else
               {
                  pos = x+1;
               }
            }

            if(pos < Meas.labelCounter && Meas.labelCounter != 0)
            {
                for(int x = 0; x<labelLength; x++)
                {
                    assignedLabels[x] = pos+x+1;
                    Meas.labelID[pos+x] = pos+x+1;
                    Meas.labelCounter = Meas.labelCounter + 1;
                }
            }
            else if(pos == Meas.labelCounter && Meas.labelCounter != 0)
            {
                int *tempBuffer = new int [Meas.labelCounter];

                for(int x = 0; x<Meas.labelCounter; x++)
                {
                    tempBuffer[x] = Meas.labelID[x];
                }

                delete [] Meas.labelID;

                Meas.labelID = new int [pos+labelLength];

                for(int x = 0; x<Meas.labelCounter; x++)
                {
                    Meas.labelID[x] = tempBuffer[x];
                }

                delete [] tempBuffer;

                for(int x = 0;  x<labelLength; x++)
                {
                    Meas.labelID[pos+x+1] = pos + x+1;
                    assignedLabels[x] = pos + x+1;
                    Meas.labelCounter = Meas.labelCounter + 1;
                }
            }
            else
            {
                Meas.labelID = new int [labelLength];
                for(int x = 0;  x<labelLength; x++)
                {
                    Meas.labelID[pos+x] = pos + x+1;
                    assignedLabels[x] = pos + x+1;
                    Meas.labelCounter = Meas.labelCounter + 1;
                }
            }

        };




        void Change_RF_Settings(float_type centerFreqIn=0, float_type sampleRateIn=0, unsigned gainIn=0, int FFTIn=0)
        {
            if(centerFreqIn != 0)
            {
                // change this to respect cases when centerFreq and sampleRate shall be changed
                SDRSetting.centerFreq = centerFreqIn;
                SDR.Change_RF_Settings(SDRSetting.centerFreq);
            }
            if(sampleRateIn != 0)
            {
                SDRSetting.sampleRate = sampleRateIn;
                SDR.Change_RF_Settings(0,SDRSetting.sampleRate);
            }
            if(gainIn != 0)
            {
                SDRSetting.gain = gainIn;
                SDR.Change_RF_Settings(0,0,SDRSetting.gain);
            }
            if(FFTIn != 0)
            {
                FFT.Change_FFT_Settings(FFTIn);
            }
        };
};


int main(int argc, char** argv)
{
    // Initialize variables
    float_type centerFreq = 1895e6;   // Carrier frequency
    float_type span =  15e6;          // Frequency span
    float_type sample = 15e6;        // Bandwidth
    double gain = 30;
    double refAmp = 0;
    int FFTlength = 16384;
    int averaging = 0;
    unsigned nTrace = 1;

    int repeat = 200*32;
    int numChannels = 2;
    int delay_ms = 0;
    
    // Create Trace object
    std::string folderName = "data/18_Slider_diff";

    // Dynamically generate folder paths for each channel
    std::vector<std::string> folderNames;
    for (int i = 0; i < numChannels; ++i) {
        folderNames.push_back(folderName + "/Channel" + std::to_string(i));
    }
    
    // Create the directories for saving data
    createDirectory(folderName);
    for (const auto& folder : folderNames) {
        EnsureFolderExists(folder); // Call the function to create folders
    }

    // Allocate stream buffers
    std::vector<int16_t*> streamBuffers(numChannels);
    for (int i = 0; i < numChannels; ++i) {
        streamBuffers[i] = new int16_t[FFTlength * 2]; // Allocate buffer
    }

    // Initialize Trace object
    Trace Trace1 = Trace(centerFreq, span, sample, gain, refAmp, FFTlength, averaging, nTrace);

    // Set buffer size and allocate a temporary StreamBuffer (if needed elsewhere)
    int bufferSize = FFTlength*2;
    // Note: This 'StreamBuffer' is allocated but never used in the loop.
    // It is also different from the 'streamBuffers' vector.
    // Consider removing it if it's not needed to avoid memory leaks.
    int16_t* StreamBuffer = new int16_t[FFTlength * 2];
    int framenumber=0;

    std::cout << "Starting in 5 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // <<< START TIMER
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Timer started. Acquiring sweeps..." << std::endl;


    // Acquire sweeps and save data
    for (int u = 0; u < 1; u++) {
        framenumber = u*repeat*span/sample;
        Trace1.Acquire_Sweep_Buffered_MultiChannel(framenumber, folderNames, FFTlength, streamBuffers, bufferSize, repeat, numChannels);
    }
    
    // <<< END TIMER & REPORT
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Loop finished." << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;


    // Clean up the buffers
    delete[] StreamBuffer; // Cleanup for the unused buffer
    for (int i = 0; i < numChannels; ++i) {
        delete[] streamBuffers[i]; // Cleanup for the vector of buffers
    }
    streamBuffers.clear();

    return 0;
}
// g++ getdata.cpp -o getdata -L/usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/libLimeSuite.so.22.09-1 -lfftw3
// ./getdata


// int main(int argc, char** argv)
// {
//     // Initialize variables
//     float_type centerFreq = 2.386e9;   // Carrier frequency
//     float_type span = 15e6;           // Frequency span
//     float_type sample = 15e6;         // Bandwidth
//     double gain = 80;
//     double refAmp = 0;
//     int FFTlength = 16384 * 16;
//     int averaging = 0;
//     unsigned nTrace = 1;

//     int repeat = 500;
//     int numChannels = 2;
//     int delay_ms = 20;

//     // Base folder name
//     std::string folderName = "data/havesignal60dbm2387direct";

//     // Dynamically generate folder paths for each channel
//     std::vector<std::string> folderNames;
//     for (int i = 0; i < numChannels; ++i) {
//         folderNames.push_back(folderName + "/Channel" + std::to_string(i));
//     }

//     // Allocate buffers for each channel
//     std::vector<int16_t*> streamBuffers(numChannels);
//     for (int i = 0; i < numChannels; ++i) {
//         streamBuffers[i] = new int16_t[FFTlength * 2]; // Allocate buffer for each channel
//     }

//     // Create Trace object
//     Trace Trace1 = Trace(centerFreq, span, sample, gain, refAmp, FFTlength, averaging, nTrace);

//     // Create the directory for saving data
//     createDirectory(folderName);

//     // Set buffer size
//     int bufferSize = FFTlength * 2;

//     // Frame number for data files
//     int framenumber = 0;

//     // Perform acquisition with parallel multi-channel processing
//     for (int u = 0; u < 1; u++) {
//         framenumber = u * repeat * span / sample;

//         // Use the parallel multi-channel function
//         Trace1.Acquire_Sweep_Buffered_MultiChannel_Parallel(
//             framenumber, folderNames, FFTlength, streamBuffers, repeat, numChannels, centerFreq);
//     }

//     // Clean up buffers
//     for (int i = 0; i < numChannels; ++i) {
//         delete[] streamBuffers[i];
//     }

//     return 0;
// }
