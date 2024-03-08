include_directories(
  ${ASMJIT_INCLUDE_DIRS}
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(viewer-bin viewer.cpp)

target_link_libraries(viewer-bin PRIVATE mitsuba nanogui ${NANOGUI_EXTRA_LIBS})

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_link_libraries(viewer-bin PRIVATE asmjit)
endif()

if (UNIX AND NOT APPLE)
  target_link_libraries(viewer-bin PRIVATE dl)
endif()

set_target_properties(viewer-bin PROPERTIES OUTPUT_NAME viewer)
