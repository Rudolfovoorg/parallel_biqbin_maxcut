cmake_minimum_required(VERSION 3.28)
project(biqbin LANGUAGES NONE)

# CMake does not build anything but uses the included MakeFile
find_program(MAKE_EXECUTABLE NAMES gmake make REQUIRED)

# Build invokes your Makefile (which copies artifacts into biqbin/)
add_custom_target(build_all ALL
  COMMAND "${MAKE_EXECUTABLE}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  USES_TERMINAL
)

# Install the Python package directory into the wheel
install(DIRECTORY "${CMAKE_SOURCE_DIR}/biqbin"
        DESTINATION "."
        FILES_MATCHING
          PATTERN "*.py"
          PATTERN "*.pyi"
          PATTERN "*.so"
          PATTERN "*.pyd"
          PATTERN "*.dll"
          PATTERN "*.dylib"
)
install(FILES
  "${CMAKE_SOURCE_DIR}/biqbin_maxcut.py"
  "${CMAKE_SOURCE_DIR}/biqbin_qubo.py"
  "${CMAKE_SOURCE_DIR}/biqbin_heuristic.py"
  DESTINATION "."
)
