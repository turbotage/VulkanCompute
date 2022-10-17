cmake_minimum_required(VERSION 3.20)

include(FetchContent)

set(avk_UseVMA ON)

FetchContent_Declare(
	avk_toolkit
	GIT_REPOSITORY		https://github.com/cg-tuwien/Auto-Vk-Toolkit.git
	GIT_TAG				fa5ac98ec87fb8c96ae47820ed9cf24c0d4aa4af
	GIT_SUBMODULES		"auto_vk"
)

FetchContent_MakeAvailable(avk_toolkit)