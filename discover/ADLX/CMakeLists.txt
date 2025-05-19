include(FetchContent)

message(STATUS "fetching ADLX git repo")
FetchContent_Declare(
  adlx
  GIT_REPOSITORY "https://github.com/GPUOpen-LibrariesAndSDKs/ADLX.git"
  GIT_TAG V1.4)
FetchContent_MakeAvailable(adlx)
message(STATUS "done fetching ADLX git repo")
