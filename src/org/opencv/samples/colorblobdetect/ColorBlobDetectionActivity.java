package org.opencv.samples.colorblobdetect;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.Window;
import android.view.WindowManager;

public class ColorBlobDetectionActivity extends Activity implements
		OnTouchListener, CvCameraViewListener {

	// Logging tag
	private static String TAG = "GDGtracker";

	// ----------------------------------------------
	// Customizable:
	private static boolean mFrontFacingLogic = true;
	private static boolean mSecondCamera = true;
	private static boolean SHOW_COORDS = false;
	private static boolean SHOW_STRENGTH = false;
	private static boolean SHOW_COLOR_SWATCH = false;
	// ----------------------------------------------

	// Other variables
	private boolean mIsColorSelected = false;
	private Mat mRgba;
	private Scalar mBlobColorRgba;
	private Scalar mBlobColorHsv;
	private ColorBlobDetector mDetector;
	private Mat mSpectrum;
	private Size SPECTRUM_SIZE;
	private int theScreenWidth;
	private int theScreenHeight;
	private float xFactor = -4.0f;
	private float yFactor = -4.0f;
	private static boolean IN_FRAME = false;
	private static String SHOW_TEXT = "";

	private CameraBridgeViewBase mOpenCvCameraView;

	// Called when this activity was paused, and then comes into focus again
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				mOpenCvCameraView.enableView();
				mOpenCvCameraView
						.setOnTouchListener(ColorBlobDetectionActivity.this);
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public ColorBlobDetectionActivity() {
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {

		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		// Choose the proper layout based on camera choice
		if (mSecondCamera)
			setContentView(R.layout.color_tracker_front_view);
		else
			setContentView(R.layout.color_blob_detection_surface_view);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);

	}

	@Override
	public void onPause() {
		// Don't keep the camera view processing when the activity has paused
		// (gone out of view)
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
		super.onPause();
	}

	@Override
	public void onResume() {
		// setup the camera view again, as activity comes
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
				mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		// Don't keep the camera view processing when the activity has been
		// destroyed
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8UC4);
		mDetector = new ColorBlobDetector();
		mSpectrum = new Mat();
		mBlobColorRgba = new Scalar(255);
		mBlobColorHsv = new Scalar(255);
		SPECTRUM_SIZE = new Size(200, 64);
	}

	public void onCameraViewStopped() {
		mRgba.release();
	}

	public boolean onTouch(View v, MotionEvent event) {
		int cols = mRgba.cols();
		int rows = mRgba.rows();

		int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
		int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

		int x = (int) event.getX() - xOffset;
		int y = (int) event.getY() - yOffset;

		// Log.i(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

		if ((x < 0) || (y < 0) || (x > cols) || (y > rows))
			return false;

		Rect touchedRect = new Rect();

		touchedRect.x = (x > 4) ? x - 4 : 0;
		touchedRect.y = (y > 4) ? y - 4 : 0;

		touchedRect.width = (x + 4 < cols) ? x + 4 - touchedRect.x : cols
				- touchedRect.x;
		touchedRect.height = (y + 4 < rows) ? y + 4 - touchedRect.y : rows
				- touchedRect.y;

		Mat touchedRegionRgba = mRgba.submat(touchedRect);

		Mat touchedRegionHsv = new Mat();
		Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv,
				Imgproc.COLOR_RGB2HSV_FULL);

		// Calculate average color of touched region
		mBlobColorHsv = Core.sumElems(touchedRegionHsv);
		int pointCount = touchedRect.width * touchedRect.height;
		for (int i = 0; i < mBlobColorHsv.val.length; i++)
			mBlobColorHsv.val[i] /= pointCount;

		mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);

		mDetector.setHsvColor(mBlobColorHsv);

		Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);

		mIsColorSelected = true;

		touchedRegionRgba.release();
		touchedRegionHsv.release();

		return false; // don't need subsequent touch events
	}

	public Mat onCameraFrame(Mat inputFrame) {
		if (IN_FRAME)
			return mRgba;
		IN_FRAME = true;

		inputFrame.copyTo(mRgba);

		if (mFrontFacingLogic)
			Core.flip(mRgba, mRgba, 1);

		if (mIsColorSelected) {
			mDetector.process(mRgba);

			if (xFactor < 0) {
				theScreenWidth = mRgba.cols();// 1280;
				theScreenHeight = mRgba.rows();// 720;

				xFactor = theScreenWidth / mDetector.smallX;
				yFactor = theScreenHeight / mDetector.smallY;

			}

			int mx = (int) (mDetector.theX * xFactor); // 4.0379
			int my = (int) (mDetector.theY * yFactor);// 4.0678

			int area = (int) (mDetector.maxArea);

			int radius = 20;// initial small scale value
			if (area > 4600) {
				radius = 60;
			} else if (area > 2700) {
				radius = 50;
			} else if (area > 1800) {
				radius = 40;
			} else if (area > 750) {
				radius = 30;
			}

			SHOW_TEXT = "";
			if (SHOW_COORDS) {
				SHOW_TEXT = "X,Y=" + mx + "," + my + " ";
			}
			if (SHOW_STRENGTH) {
				SHOW_TEXT += "Area=" + area;
			}
			if (SHOW_COORDS || SHOW_STRENGTH) {
				Core.putText(mRgba, SHOW_TEXT, new Point(50, 100), 3, 1,
						new Scalar(255, 0, 0, 255), 2);
			}

			Core.circle(mRgba, new Point(mx, my), radius + 4, new Scalar(255,
					255, 255, 255), 6);
			Core.circle(mRgba, new Point(mx, my), radius + 2, new Scalar(0, 0,
					0, 0), 3);

			Core.circle(mRgba, new Point(mx, my), radius, new Scalar(
					mBlobColorRgba.val[0], mBlobColorRgba.val[1],
					mBlobColorRgba.val[2], mBlobColorRgba.val[3]), -1);

			if (SHOW_COLOR_SWATCH) {
				Mat colorLabel = mRgba.submat(4, 68, 4, 68);
				colorLabel.setTo(mBlobColorRgba);

				Mat spectrumLabel = mRgba.submat(4, 4 + mSpectrum.rows(), 70,
						70 + mSpectrum.cols());
				mSpectrum.copyTo(spectrumLabel);
			}
		}

		IN_FRAME = false;

		return mRgba;
	}

	private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
		Mat pointMatRgba = new Mat();
		Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
		Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL,
				4);

		return new Scalar(pointMatRgba.get(0, 0));
	}
}