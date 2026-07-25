#include "KeyPressDetector.h"

#ifdef _MSC_VER
#include <conio.h>

char GGL::KeyPressDetector::GetPressedChar() {
	return _getch();
}

char GGL::KeyPressDetector::GetPressedCharNonBlocking() {
	return _kbhit() ? (char)_getch() : 0;
}

#else
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

char GGL::KeyPressDetector::GetPressedCharNonBlocking() {
	// Configure stdin for non-blocking, unbuffered single-char reads.
	struct termios oldt = { 0 };
	if (tcgetattr(0, &oldt) < 0)
		return 0;
	struct termios newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	newt.c_cc[VMIN] = 0;
	newt.c_cc[VTIME] = 0;
	tcsetattr(0, TCSANOW, &newt);

	int oldFlags = fcntl(0, F_GETFL, 0);
	fcntl(0, F_SETFL, oldFlags | O_NONBLOCK);

	char buf = 0;
	if (read(0, &buf, 1) <= 0)
		buf = 0;

	// Restore previous terminal/fd state.
	fcntl(0, F_SETFL, oldFlags);
	tcsetattr(0, TCSANOW, &oldt);
	return buf;
}

char GGL::KeyPressDetector::GetPressedChar() {
	// https://stackoverflow.com/questions/421860/capture-characters-from-standard-input-without-waiting-for-enter-to-be-pressed
	char buf = 0;
	struct termios old = { 0 };
	if (tcgetattr(0, &old) < 0)
		perror("tcsetattr()");
	old.c_lflag &= ~ICANON;
	old.c_lflag &= ~ECHO;
	old.c_cc[VMIN] = 1;
	old.c_cc[VTIME] = 0;
	if (tcsetattr(0, TCSANOW, &old) < 0)
		perror("tcsetattr ICANON");
	if (read(0, &buf, 1) < 0)
		perror("read()");
	old.c_lflag |= ICANON;
	old.c_lflag |= ECHO;
	if (tcsetattr(0, TCSADRAIN, &old) < 0)
		perror("tcsetattr ~ICANON");
	return (buf);
}

#endif