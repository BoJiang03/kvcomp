#ifndef TCOM_UTILS_H
#define TCOM_UTILS_H

#include <iostream>
#include <cstdlib>

#define tcom_assert(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "TCom Assertion failed: " << msg << "\n" \
                      << "File: " << __FILE__ << "\n" \
                      << "Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

#endif //TCOM_UTILS_H