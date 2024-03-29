package com.arun.falldetection;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.CountDownTimer;
import android.os.Environment;
import android.provider.Telephony;
import android.speech.tts.TextToSpeech;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.telephony.SmsManager;
import android.text.InputType;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final int N_SAMPLES = 200;
    private static final int PERMISSION_REQUEST_CODE = 1;
    private static final String TAG = "MainActivity: " ;
    private static final double ACC_THRESHOLD = 23;
    private static List<Float> x, y, z;

    private TextView downstairsTv, joggingTv, sittingTv, standingTv, upstairsTv, walkingTv, falling, status;
    private Button action_btn, settings_btn;

    private float[] results;

    private TensorFlowClassifier classifier;

    private String[] labels = {"Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking", "fall"};

    private String mText;

    private boolean fallDetected = false;
    private AlertDialog countdown;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT >= 23) {
            if (checkPermission()) {
                Log.e("permission", "Permission already granted.");
            } else {
                requestPermission();
            }
        }

        x = new ArrayList<>();
        y = new ArrayList<>();
        z = new ArrayList<>();

        downstairsTv = findViewById(R.id.downstairs_prob);
        joggingTv = findViewById(R.id.jogging_prob);
        sittingTv = findViewById(R.id.sitting_prob);
        standingTv = findViewById(R.id.standing_prob);
        upstairsTv = findViewById(R.id.upstairs_prob);
        walkingTv = findViewById(R.id.walking_prob);
        status = findViewById(R.id.status);
        falling = findViewById(R.id.fall_prob);

        classifier = new TensorFlowClassifier(getApplicationContext());

        action_btn = findViewById(R.id.activity_btn);
        action_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String btn_text = "Start Monitoring";
                if (action_btn.getText().equals(btn_text)) {
                    action_btn.setText("Stop Monitoring");
//                    fallDetected();
                    getSensorManager().registerListener(MainActivity.this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME);
                } else {
                    action_btn.setText("Start Monitoring");
                    getSensorManager().unregisterListener(MainActivity.this);
                }

            }
        });

        settings_btn = findViewById(R.id.config_btn);
        settings_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("Emergency SMS number:");

                final EditText input = new EditText(MainActivity.this);
                input.setInputType(InputType.TYPE_CLASS_PHONE);
                builder.setView(input);

                builder.setPositiveButton("Set", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        mText = input.getText().toString();
                        Log.d(TAG, "onClick: "+mText);
                    }
                });
                builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.cancel();
                    }
                });
                builder.show();
            }
        });
    }

    private boolean checkPermission() {
        int result = ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS);
        if (result == PackageManager.PERMISSION_GRANTED) {
            return true;
        } else {
            return false;
        }
    }

    private void requestPermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.SEND_SMS}, PERMISSION_REQUEST_CODE);
    }

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            activityPrediction();
            x.add(event.values[0]);
            y.add(event.values[1]);
            z.add(event.values[2]);
        }
    }

    private void fallDetected() {

        countdown = new AlertDialog.Builder(this)
                .setTitle("Have you fallen? Please Respond!")
                .setMessage("SMS will be sent automatically in 10 seconds")
                .setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        sendSMS();
                    }
                })
                .setNegativeButton("No", null)
                .create();
        countdown.setOnShowListener(new DialogInterface.OnShowListener() {
            private static final int SMS_MILLIS = 10000;
            @Override
            public void onShow(final DialogInterface dialog) {
                final Button defaultButton = countdown.getButton(AlertDialog.BUTTON_POSITIVE);
                final CharSequence positiveButtonText = defaultButton.getText();
                new CountDownTimer(SMS_MILLIS, 100){

                    @Override
                    public void onTick(long millisUntilFinished) {
                        defaultButton.setText(String.format(
                                Locale.getDefault(), "%s (%d)",
                                positiveButtonText,
                                TimeUnit.MILLISECONDS.toSeconds(millisUntilFinished)
                        ));
                    }

                    @Override
                    public void onFinish() {
                        if (countdown.isShowing()) {
                            countdown.setMessage("SMS Sent!");
                            sendSMS();
                        }
                    }
                }.start();
            }
        });
        countdown.show();


//        countdown = new AlertDialog.Builder(this)
//                .setTitle("Have you fallen? Please Respond!")
//                .setMessage("SMS will be sent automatically in 10 seconds")
//                .setNegativeButton("No, I'm alright.",
//                        new DialogInterface.OnClickListener() {
//                            @Override
//                            public void onClick(DialogInterface dialog, int which) {
//                                dialog.cancel();
//                            }
//                        });
//
//        final AlertDialog alert = countdown.create();
//        alert.show();
//
//        final Timer timer = new Timer();
//        timer.schedule(new TimerTask() {
//            @Override
//            public void run() {
//                timer.cancel();
//                sendSMS();
//            }
//        }, 10000);
    }

    private boolean isFallDetected(double x, double y, double z) {
        double acceleration = this.calculateSumVector(x, y, z);
        if (acceleration > ACC_THRESHOLD) {
            return true;
        }
        return false;
    }

    private double calculateSumVector(double x, double y, double z) {
        return Math.abs(x) + Math.abs(y) + Math.abs(z);
    }

    private void activityPrediction() {
        if (x.size() == N_SAMPLES && y.size() == N_SAMPLES && z.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();
            data.addAll(x);
            data.addAll(y);
            data.addAll(z);

            results = classifier.predictProbabilities(toFloatArray(data));

            if (results == null || results.length == 0) {
                return;
            } else {
                float max = -1;
                int idx = -1;
                for (int i = 0; i < results.length; i++) {
                    if (results[i] > max) {
                        idx = i;
                        max = results[i];
                    }
                }
                status.setText(labels[idx]);
            }

            downstairsTv.setText("Downstairs: " + Float.toString(round(results[0], 2)));
            joggingTv.setText("Jogging: " + Float.toString(round(results[1], 2)));
            sittingTv.setText("Sitting: " + Float.toString(round(results[2], 2)));
            standingTv.setText("Standing: " + Float.toString(round(results[3], 2)));
            upstairsTv.setText("Upstairs: " + Float.toString(round(results[4], 2)));
            walkingTv.setText("Walking: " + Float.toString(round(results[5], 2)));
            falling.setText("Fall: " + Float.toString(round(results[6], 2)));



            if ((round(results[6], 2)) > 0.2){
                fallDetected();
            }

            x.clear();
            y.clear();
            z.clear();
        }
    }

    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list){
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private static float round(float result, int i) {
        BigDecimal bd = new BigDecimal(Float.toString(result));
        bd = bd.setScale(i, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    public void sendSMS() {
        Log.d(TAG, "sendSMS: started");
        Log.d(TAG, "sendSMS: "+mText);

        try {
            SmsManager smsManager = SmsManager.getDefault();
            smsManager.sendTextMessage(mText, null, "Help! I've fallen!!", null, null);
            Toast.makeText(this, "SMS Sent!", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "sendSMS: sent!");
        } catch (Exception e) {
            Log.d(TAG, "sendSMS: failed to send!");
            Toast.makeText(this, "Failed to send sms!", Toast.LENGTH_SHORT).show();
        }
    }


    @Override
    protected void onPause() {
        getSensorManager().unregisterListener(this);
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }
}
