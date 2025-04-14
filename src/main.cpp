/* main.cpp */

#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <deque>
#include <csignal>
#include <chrono>
#include <sched.h>
#include <string.h>
#include <sys/statvfs.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iomanip>
#include "rp.h"
#include "Common.hpp"
#include "SystemUtils.hpp"
#include "DataAcquisition.hpp"
#include "DataWriterCSV.hpp"
#include "DataWriterDAC.hpp"
#include "ModelProcessing.hpp"
#include "ModelWriterCSV.hpp"
#include "ModelWriterDAC.hpp"
#include "DAC.hpp"

bool save_data_csv = false;
bool save_data_dac = false;
bool save_output_csv = false;
bool save_output_dac = false;

int main()
{
    if (rp_Init() != RP_OK)
    {
        std::cerr << "Rp API init failed!" << std::endl;
        return -1;
    }

    std::signal(SIGINT, signal_handler);

    folder_manager("DataOutput");
    folder_manager("ModelOutput");

    std::cout << "Starting program" << std::endl;

    if (!ask_user_preferences(save_data_csv, save_data_dac, save_output_csv, save_output_dac))
    {
        std::cerr << "User input failed. Exiting." << std::endl;
        return -1;
    }
    ::save_data_csv = save_data_csv;
    ::save_data_dac = save_data_dac;
    ::save_output_csv = save_output_csv;
    ::save_output_dac = save_output_dac;

    initialize_acq();
    initialize_DAC();
    std::thread acq_thread1(acquire_data, std::ref(channel1), RP_CH_1);
    std::thread acq_thread2(acquire_data, std::ref(channel2), RP_CH_2);
    std::thread model_thread1(model_inference, std::ref(channel1));
    std::thread model_thread2(model_inference, std::ref(channel2));

    std::thread write_thread_csv1, write_thread_dac1, log_thread_csv1, log_thread_dac1;
    std::thread write_thread_csv2, write_thread_dac2, log_thread_csv2, log_thread_dac2;

    if (save_data_csv)
        write_thread_csv1 = std::thread(write_data_csv, std::ref(channel1), "DataOutput/data_ch1.csv");
    write_thread_csv2 = std::thread(write_data_csv, std::ref(channel2), "DataOutput/data_ch2.csv");
    if (save_data_dac)
        write_thread_dac1 = std::thread(write_data_dac, std::ref(channel1), RP_CH_1);
    write_thread_dac2 = std::thread(write_data_dac, std::ref(channel2), RP_CH_2);

    if (save_output_csv)
        log_thread_csv1 = std::thread(log_results_csv, std::ref(channel1), "ModelOutput/output_ch1.csv");
    log_thread_csv2 = std::thread(log_results_csv, std::ref(channel2), "ModelOutput/output_ch2.csv");
    if (save_output_dac)
        log_thread_dac1 = std::thread(log_results_dac, std::ref(channel1), RP_CH_1);
    log_thread_dac2 = std::thread(log_results_dac, std::ref(channel2), RP_CH_2);

    // set_thread_priority(acq_thread1, acq_priority);
    // set_thread_priority(acq_thread2, acq_priority);
    // set_thread_priority(write_thread_csv1, write_csv_priority);
    // set_thread_priority(write_thread_cs2, write_csv_priority);
    // set_thread_priority(write_thread_dac1, write_dac_priority);
    // set_thread_priority(write_thread_dac2, write_dac_priority);
    set_thread_priority(model_thread1, model_priority);
    set_thread_priority(model_thread2, model_priority);
    // set_thread_priority(log_thread_csv1, log_csv_priority);
    // set_thread_priority(log_thread_csv2, log_csv_priority);
    // set_thread_priority(log_thread_dac1, log_dac_priority);
    // set_thread_priority(log_thread_dac2, log_dac_priority);

    if (acq_thread1.joinable())
        acq_thread1.join();
    if (acq_thread2.joinable())
        acq_thread2.join();
    if (model_thread1.joinable())
        model_thread1.join();
    if (model_thread2.joinable())
        model_thread2.join();
    if (save_data_csv && write_thread_csv1.joinable())
        write_thread_csv1.join();
    if (save_data_csv && write_thread_csv2.joinable())
        write_thread_csv2.join();
    if (save_data_dac && write_thread_dac1.joinable())
        write_thread_dac1.join();
    if (save_data_dac && write_thread_dac2.joinable())
        write_thread_dac2.join();
    if (save_output_csv && log_thread_csv1.joinable())
        log_thread_csv1.join();
    if (save_output_csv && log_thread_csv2.joinable())
        log_thread_csv2.join();
    if (save_output_dac && log_thread_dac1.joinable())
        log_thread_dac1.join();
    if (save_output_dac && log_thread_dac2.joinable())
        log_thread_dac2.join();

    cleanup();
    print_channel_stats(channel1);
    print_channel_stats(channel2);

    return 0;
}
