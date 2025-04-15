/*SystemUtils.cpp*/

#include "SystemUtils.hpp"
#include <iostream>
#include <csignal>
#include <iomanip>
#include <filesystem>
#include <sys/statvfs.h>
#include <pthread.h>

volatile std::sig_atomic_t interrupted = 0;

bool is_disk_space_below_threshold(const char *path, double threshold)
{
    struct statvfs stat;
    if (statvfs(path, &stat) != 0)
    {
        std::cerr << "Error getting filesystem statistics." << std::endl;
        return false;
    }

    double available_space = stat.f_bsize * stat.f_bavail;
    return available_space < threshold;
}

bool set_thread_priority(std::thread &th, int priority)
{
    struct sched_param param;
    param.sched_priority = priority;

    if (pthread_setschedparam(th.native_handle(), SCHED_FIFO, &param) != 0)
    {
        std::cerr << "Failed to set thread priority to " << priority << std::endl;
        return false;
    }
    return true;
}

bool set_thread_affinity(std::thread &th, int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset) != 0)
    {
        return false;
    }
    return true;
}

void signal_handler(int sig)
{
    if (sig == SIGINT)
    {
        interrupted = 1;
        std::cout << "\n^CSIGINT received, initiating graceful shutdown..." << std::endl;

        stop_program.store(true);
        stop_acquisition.store(true);

        std::cin.setstate(std::ios::failbit);

        channel1.cond_write_csv.notify_all();
        channel1.cond_write_dac.notify_all();
        channel1.cond_model.notify_all();
        channel1.cond_log_csv.notify_all();
        channel1.cond_log_dac.notify_all();

        channel2.cond_write_csv.notify_all();
        channel2.cond_write_dac.notify_all();
        channel2.cond_model.notify_all();
        channel2.cond_log_csv.notify_all();
        channel2.cond_log_dac.notify_all();
    }
}

void print_duration(const std::string &label, uint64_t start_ns, uint64_t end_ns)
{
    auto duration_ns = end_ns > start_ns ? end_ns - start_ns : 0;
    auto duration_ms = duration_ns / 1'000'000;

    auto minutes = duration_ms / 60000;
    auto seconds = (duration_ms % 60000) / 1000;
    auto ms = duration_ms % 1000;

    std::cout << std::left << std::setw(40) << label + " acquisition time:"
              << minutes << " min " << seconds << " sec " << ms << " ms\n";
}

void print_channel_stats(const Channel &channel)
{
    std::cout << "====================================\n\n";

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        channel.end_time_point - channel.trigger_time_point);

    auto minutes = duration.count() / 60000;
    auto seconds = (duration.count() % 60000) / 1000;
    auto ms = duration.count() % 1000;

    std::cout << "Total acquisition time for Channel " << channel.channel_id + 1 << ": "
              << minutes << " minutes "
              << seconds << " seconds "
              << ms << " milliseconds\n";

    std::cout << std::left << std::setw(60) << "Total data acquired:" << channel.acquire_count.load() << '\n';
    if (save_data_csv)
    {
        std::cout << std::left << std::setw(60) << "Total lines written to csv file:" << channel.write_count_csv.load() << '\n';
    }
    if (save_data_dac)
    {
        std::cout << std::left << std::setw(60) << "Total lines written to dac:" << channel.write_count_csv.load() << '\n';
    }
    std::cout << std::left << std::setw(60) << "Total model calculated:" << channel.model_count.load() << '\n';
    if (save_output_csv)
    {
        std::cout << std::left << std::setw(60) << "Total results logged to CSV file:" << channel.log_count_csv.load() << '\n';
    }
    if (save_output_dac)
    {
        std::cout << std::left << std::setw(60) << "Total results written to DAC:" << channel.log_count_dac.load() << '\n';
    }

    std::cout << "\n====================================\n";
}

void folder_manager(const std::string &folder_path)
{
    namespace fs = std::filesystem;

    try
    {
        fs::path dir_path(folder_path);

        if (fs::exists(dir_path))
        {
            for (const auto &entry : fs::directory_iterator(dir_path))
            {
                try
                {
                    fs::remove_all(entry);
                }
                catch (const fs::filesystem_error &e)
                {
                    std::cerr << "Failed to delete file: " << entry.path() << " - " << e.what() << std::endl;
                }
            }
        }
        else
        {
            if (!fs::create_directories(dir_path))
            {
                std::cerr << "Failed to create directory: " << folder_path << std::endl;
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}

bool ask_user_preferences(bool &save_data_csv, bool &save_data_dac, bool &save_output_csv, bool &save_output_dac)
{
    int max_attempts = 3;

    for (int attempt = 1; attempt <= max_attempts; ++attempt)
    {
        if (interrupted)
            return false;

        int save_choice;
        std::cout << "Do you want to save acquired data?\n"
                  << " 1. As CSV only\n"
                  << " 2. To DAC only\n"
                  << " 3. Both CSV and DAC\n"
                  << " 4. None\n"
                  << "Enter your choice (1-4): ";
        std::cin >> save_choice;

        if (std::cin.fail() || interrupted)
        {
            std::cerr << "Input interrupted or invalid. Aborting...\n";
            return false;
        }

        if (save_choice >= 1 && save_choice <= 4)
        {
            save_data_csv = (save_choice == 1 || save_choice == 3);
            save_data_dac = (save_choice == 2 || save_choice == 3);
            break;
        }
        else
        {
            std::cerr << "Invalid input. Please enter a number between 1 and 4.\n";
            if (attempt == max_attempts)
                return false;
        }
    }

    for (int attempt = 1; attempt <= max_attempts; ++attempt)
    {
        if (interrupted)
            return false;

        int output_option;
        std::cout << "\nChoose what to do with model output:\n"
                  << " 1. Save as CSV only\n"
                  << " 2. Output to DAC only\n"
                  << " 3. Both CSV and DAC\n"
                  << " 4. None\n"
                  << "Enter your choice (1-4): ";
        std::cin >> output_option;

        if (std::cin.fail() || interrupted)
        {
            std::cerr << "Input interrupted or invalid. Aborting...\n";
            return false;
        }

        if (output_option >= 1 && output_option <= 4)
        {
            save_output_csv = (output_option == 1 || output_option == 3);

            if (save_data_dac && (output_option == 2 || output_option == 3))
            {
                save_output_dac = false;
                std::cerr << "\n[Warning] DAC is already used for saving raw data.\n"
                          << "Model output will NOT be sent to DAC.\n";
            }
            else
            {
                save_output_dac = (output_option == 2 || output_option == 3);
            }

            return true;
        }
        else
        {
            std::cerr << "Invalid input. Please enter a number between 1 and 4.\n";
            if (attempt == max_attempts)
                return false;
        }
    }

    return true;
}
