#pragma once

#include "../Framework.h"

namespace GGL {
	namespace KeyPressDetector {
		// Blocks until a key is pressed and returns it.
		char GetPressedChar();

		// Non-blocking: returns the pressed key, or 0 if no key is currently available.
		// Lets the training loop poll for the quit key without a dedicated blocking thread,
		// enabling a clean shutdown (no exit(0), no dangling detached thread).
		char GetPressedCharNonBlocking();
	}
}