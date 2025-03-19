cmake_minimum_required(VERSION 3.1)


SET(FT_DIR /home/heaven7/heaven7/libs/FreeType/freetype-2.13.3)
include_directories(${FT_DIR}/include)
SET(FT_LIBS ${FT_DIR}/lib/libfreetype.so)
###

file(GLOB_RECURSE SRC_PIL_RESIZE
    *.h
    *.hpp
    *.cpp
    *.c
    *.cc
    )

add_executable(resize ${SRC_PIL_RESIZE})

target_link_libraries(
  resize
  PUBLIC # Opencv libraries
         "${OpenCV_LIBS}"
         # CLI
         CLI11::CLI11
         # magic_enum
         magic_enum::magic_enum
         # PillowResize lib
         PillowResize
    ${FT_LIBS}
)

# Define compile features: c++17 and no extensions
target_compile_features(resize PRIVATE cxx_std_17)
set_target_properties(resize PROPERTIES CXX_EXTENSIONS OFF)

set(EXECUTABLES ${CMAKE_CURRENT_BINARY_DIR}/resize)

# Install binaries into the system
include(GNUInstallDirs)
install(
  FILES ${EXECUTABLES}
  DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}/${PROJECT_NAME}
  PERMISSIONS WORLD_EXECUTE WORLD_WRITE WORLD_READ)
