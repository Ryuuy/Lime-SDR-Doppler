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
        lms_stream_t streamId;

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
            //open the first device
            lms_info_str_t list[8];
            int n;

            if ((n = LMS_GetDeviceList(list)) < 0)  //NULL can be passed to only get number of devices
                SDRerror("No devices found", device);

            if (LMS_Open(&device, list[0], NULL))
                SDRerror("Device could not be opened", device);

            const char *config_file = "./1010.ini";
            if (LMS_LoadConfig(device, config_file) == 0) {
                printf("配置文件加载成功\n");
            } else {
                printf("配置文件加载失败\n");
            }
            // if (LMS_SetLOFrequency(device, LMS_CH_RX, 0, centerFreq) != 0)
            //     SDRerror("Center Frequency could not be set", device);

            // if(LMS_SetGaindB(device, LMS_CH_RX, 0, gain)!=0)
            //     SDRerror("Gain could not be set", device);

            // Start the stream
            Setup_Stream();
            Start_Stream();

            return 1;
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

        int Setup_Stream()
        {
            streamId.channel = 0;               //channel number
            streamId.fifoSize = 1024 * 1024;    //fifo size in samples
            streamId.throughputVsLatency = 1.0; //optimize for max throughput
            streamId.isTx = false;              //RX channel
            streamId.dataFmt = lms_stream_t::LMS_FMT_I12;     //12-bit integers //lms_stream_t::

            //Setup Stream
            if (LMS_SetupStream(device, &streamId) != 0)
                SDRerror("Stream could not be set up", device);

            return 1;
        }

        void Start_Stream()
        {
            //Start streaming
            LMS_StartStream(&streamId);
        }

        void Stop_Stream()
        {
            //Stop streaming
            LMS_StopStream(&streamId);
        }

        void Close_Stream()
        {
            LMS_DestroyStream(device, &streamId);
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
            SDRSetting.sampleRate = resBWIn*2;
            // Calculate time for SDR to sample a frame
            SDRSetting.frameTime = 1/SDRSetting.sampleRate*FFTIn;
            // Set initial Center Frequency to beginning of span. Note: the term Center Frequency is misleading here as the frequency set as "Center Frequency"
            // in the SDR will from where the downmixed Basband signal starts. ->Start Frequency is a better name here
            SDRSetting.centerFreq = centerFreqIn - dispSpanIn / 2;//  starting freq actually
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
        

        void Aqcuire_Sweep_Buffered(int frameCounter, const std::string& folderName, int FFTLength, int16_t* StreamBuffer, int bufferSize) 
        {
            // 创建文件路径并以二进制模式打开文件
            std::ostringstream fileName;
            fileName << folderName << "/sdr_data_frame_" << frameCounter << ".bin";
            std::ofstream outFile(fileName.str(), std::ios::binary);  // 二进制模式

            if (!outFile.is_open()) {
                std::cerr << "Failed to open file for writing!" << std::endl;
                return;
            }

            // 每个文件保存一帧数据
            for (int z = 0; z < (int)ceil(Display.span / Display.resBW); z++) {
                Change_RF_Settings(Display.centerFreq+SDRSetting.sampleRate*z);
                // 从 LimeSDR 接收样本
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                // receive samples from SDR

                int samplesRead = LMS_RecvStream(&SDR.streamId, StreamBuffer, FFTLength, NULL, 3000);
                Change_RF_Settings(Display.centerFreq+SDRSetting.sampleRate*z);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                samplesRead = LMS_RecvStream(&SDR.streamId, StreamBuffer, FFTLength, NULL, 3000);
                

                // 将 StreamBuffer 中的样本直接写入文件，samplesRead 是样本对数，需要写入 samplesRead * 2 个 int16
                outFile.write(reinterpret_cast<const char*>(StreamBuffer), samplesRead * 2 * sizeof(int16_t));
            }

            // 关闭文件
            outFile.close();
            std::cout << "Frame " << frameCounter << " saved to " << fileName.str() << std::endl;
            // std::this_thread::sleep_for(std::chrono::milliseconds(9));
        }






        //**************************************************************************************

        void setRefAmp()
        {
            GnuplotHandle.set_yrange(Display.refAmp-130, Display.refAmp);
        };

        //**************************************************************************************
        void Plot_Spectrum()
        {
            FILE* plotFile;
            int FFTLength = getFFTLength();
            // set upper limit of plot to refAmp
            //GnuplotHandle.set_yrange(Display.refAmp-100, Display.refAmp);

            // write plotFile from contents of XAxisBuffer and YAxisBuffer

            plotFile = fopen("plotFile.txt", "w+");
            // if (plotFile == NULL) {
            //         printf("plotFile is NULL\n");
            //     } else {
            //         printf("plotFile is not NULL\n");
            //     }

            for(int x=0; x<(int)ceil(Display.span/Display.resBW)*FFTLength/2; x++)
            {
               // printf("Debug: XAxisBuffer[%d] = %f, YAxisBuffer[%d] = %f\n", x, XAxisBuffer[x], x, YAxisBuffer[x]);
                fprintf(plotFile, "%f   %f\n", XAxisBuffer[x], YAxisBuffer[x]);
                //printf("%f   %f\n", XAxisBuffer[x], YAxisBuffer[x]);
            }
            fclose(plotFile);

            GnuplotHandle.cmd("plot 'plotFile.txt' with lines linestyle 1");

            // set bounding box for channel measurement


            //GnuplotHandle.cmd("set object 2 rect from 950000000,-150 to 970000000,-15 behind fillcolor rgb 'red' fillstyle solid 0.5 border");
            //clear last plot before making a new one
            GnuplotHandle.reset_plot();
        };

        int Change_Detector_Type(char* setting)
        {
            if(strcmp(setting, "ClearWrite") == 0)
            {
                Detector.clearWrite = true;
                Detector.maxHold = false;
                Detector.minHold = false;
                return 1;
            }
            else if(strcmp(setting, "MaxHold") == 0)
            {
                Detector.clearWrite = false;
                Detector.maxHold = true;
                Detector.minHold = false;
                return 1;
            }
            else if(strcmp(setting, "MinHold") == 0)
            {
                Detector.clearWrite = false;
                Detector.maxHold = false;
                Detector.minHold = true;
                return 1;
            }
            else
            {           
                inputError("Unallowed Detector setting.\n");
                return 0;
            }
        };


        int Set_Markers(float* markerIn)
        {
            if(Meas.markerLabelID == NULL)
            {
                // create double pointer to point to a two dimensional array
                Meas.markerLabelID = new int* [3];

                for(int x = 0; x<3; x++)
                {
                    Meas.markerLabelID[x] = new int [2];
                    Meas.markerLabelID[x][0] = 0;
                    Meas.markerLabelID[x][1] = 0;
                }
            }
            // Set markers (if markers are all 0 -> display no markers)
            for(int x = 0; x<3; x++)
            {
                if(markerIn[x] > 0)
                {
                    if(markerIn[x] >= (Display.centerFreq - (Display.span/2)) && markerIn[x] <= (Display.centerFreq + (Display.span/2)))
                    {
                        assignLabelID(Meas.markerLabelID[x], 2);
                        Meas.marker[x] = markerIn[x];
                        Meas.markerLabelPos[x][0] = Display.centerFreq + (Display.span/2) - 2.2*(Display.span/10);
                        Meas.markerLabelPos[x][1] = Display.refAmp - 5 - (5 * (x+1));
                    }
                    else
                    {
                        char buffer[33];
                        sprintf(buffer, "Marker %i not in Span boundaries\n", x);
                        inputError(buffer);
                        return 0;
                    }
                }
            }
            return 1;
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
                SDRSetting.centerFreq = centerFreqIn - SDRSetting.sampleRate/4;
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


//Main Function
int main(int argc, char** argv)
{
    // Initialize variables
    float_type centerFreq = 1.010e9;   // Carrier frequency
    float_type span = 45e6;          // Frequency span
    float_type sample = 15e6;        // Bandwidth
    double gain = 80;
    double refAmp = 0;
    int FFTlength = 16384;
    int averaging = 0;
    unsigned nTrace = 1;

    // Create Trace object
    Trace Trace1 = Trace(centerFreq, span, sample, gain, refAmp, FFTlength, averaging, nTrace);
    Trace1.setRefAmp();

    // Create the directory for saving data
    std::string folderName = "45Mdatacalibration";
    createDirectory(folderName);

    // Set buffer size and allocate StreamBuffer
    int bufferSize = FFTlength*2;  // Adjust buffer size based on performance needs
    int16_t* StreamBuffer = new int16_t[FFTlength * 2];  // Temporary buffer for I/Q data

    // Acquire 10000 sweeps and save data
    for (int u = 0; u < 10000; u++) {
        Trace1.Aqcuire_Sweep_Buffered(u, folderName, FFTlength, StreamBuffer, bufferSize);  // Pass the required parameters
    }

    // Clean up the buffer
    delete[] StreamBuffer;

    return 0;
}

