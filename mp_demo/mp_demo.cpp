////////////////////////////////////////////////////////////////////////////
// Русский
#include <stdlib.h>

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Button.H>
#include <FL/fl_ask.H>
#include <FL/fl_draw.H>

#include "cuda_core.h"

void callback1(Fl_Widget *w, void *) {
	int r=system("1.bat");
}

void callback2(Fl_Widget *w, void *) {
	int r=system("2.bat &");
}

void callback1(Fl_Widget *w, void *) {
	int r=system("3.bat");
}

void callback1(Fl_Widget *w, void *) {
	int r=system("4.bat");
}

void callback1(Fl_Widget *w, void *) {
	int r=system("5.bat");
}


void callback_about(Fl_Widget *w, void *) {
	fl_message("LaBSim-Flow\nLatice Boltzman simulation flow\nhttp://ancient.hydro.nsc.ru/labsimflow");
}

class Win:public Fl_Window{
public:
	Win():Fl_Window(800, 300, "LaBSim-Flow"){
		Fl_Menu_Bar * menubar=new Fl_Menu_Bar(0,0,w(),30);
		//	Fl_Window::resizable(canvas);
		end();
		menubar->add("Multi components",          FL_ALT+'1', callback1, 0, 0);
		menubar->add("Phase transition",          FL_ALT+'2', callback2, 0, 0);
		menubar->add("Heat transfer",             FL_ALT+'3', callback3, 0, 0);
		menubar->add("Movable cellular automata", FL_ALT+'4', callback4, 0, 0);		
		menubar->add("Simple LBE",                FL_ALT+'5', callback5, 0, 0);
		menubar->add("About",                     FL_ALT+'a', callback_about, 0, 0);
		show(); 
	}
	void draw(){
		Fl_Window::draw();		
		gpu_info();
		fl_draw(str_info.c_str(), 10, 40, 600, 300, FL_ALIGN_TOP_LEFT, 0, '@');
	}
};

int main(){
	Win win;        // создаем окно
	
	return Fl::run();                     // запускаем обработчик событий
}

