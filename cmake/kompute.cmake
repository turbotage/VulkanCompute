cmake_minimum_required(VERSION 3.20)

include(FetchContent)

FetchContent_Declare(
	kompute
	GIT_REPOSITORY		https://github.com/KomputeProject/kompute
	GIT_TAG				e8f051fc6695aaff429a025727cb116e3377aea2
)

FetchContent_MakeAvailable(kompute)