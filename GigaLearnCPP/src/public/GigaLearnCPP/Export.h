#pragma once

// Minimal DLL import/export macro, kept dependency-free so lightweight public headers
// (e.g. the checkpoint SDK) can be exported without pulling in the whole framework.
// Framework.h defines the same macros; the include guards below keep them consistent.

#ifndef RG_EXPORTED
	#if defined(_MSC_VER)
		#define RG_EXPORTED __declspec(dllexport)
		#define RG_IMPORTED __declspec(dllimport)
	#else
		#define RG_EXPORTED __attribute__((visibility("default")))
		#define RG_IMPORTED
	#endif
#endif

#ifndef RG_IMEXPORT
	#ifdef WITHIN_GGL
		#define RG_IMEXPORT RG_EXPORTED
	#else
		#define RG_IMEXPORT RG_IMPORTED
	#endif
#endif
