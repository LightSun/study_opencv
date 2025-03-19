#ifndef CVX_TEXT_H
#define CVX_TEXT_H

#include <ft2build.h>  
#include FT_FREETYPE_H  
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
class CvxText
{
	CvxText& operator=(const CvxText&);
public:
	CvxText(const char *freeType);
	virtual ~CvxText();


	void getFont(int *type,
		CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);


	void setFont(int *type,
		CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);


	void restoreFont();

	int putText(cv::Mat &frame, const char    *text, CvPoint pos);


	int putText(cv::Mat &frame, const wchar_t *text, CvPoint pos);


	int putText(cv::Mat &frame, const char    *text, CvPoint pos, CvScalar color);


	int putText(cv::Mat &frame, const wchar_t *text, CvPoint pos, CvScalar color);

	//================================================================  
	//================================================================  

private:


	void putWChar(cv::Mat &frame, wchar_t wc, CvPoint &pos, CvScalar color);

	//================================================================  
	//================================================================  

private:

	FT_Library   m_library;   
	FT_Face      m_face;      

	

	int         m_fontType;
	CvScalar   m_fontSize;
	bool      m_fontUnderline;
	float      m_fontDiaphaneity;
};

#endif


