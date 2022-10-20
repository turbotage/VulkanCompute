cmake_minimum_required(VERSION 3.20)

include(FetchContent)

set(avk_UseVMA ON)

#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

#set(avk_toolkit_LibraryType SHARED)

FetchContent_Declare(
	avk_toolkit
	GIT_REPOSITORY		https://github.com/cg-tuwien/Auto-Vk-Toolkit.git
	GIT_TAG				fa5ac98ec87fb8c96ae47820ed9cf24c0d4aa4af
	GIT_SUBMODULES		"auto_vk"
)

FetchContent_MakeAvailable(avk_toolkit)

if (MSVC)
	target_compile_options(Auto_Vk_Toolkit PUBLIC /Zc:preprocessor)
endif(MSVC)

set_property(TARGET Auto_Vk_Toolkit PROPERTY CXX_STANDARD 20)