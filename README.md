# Face-Recognition-Attendance-System-Using-LPBH


## Overview
This project is a **Face Recognition Attendance System** built using **Python, OpenCV, SQLite, and Tkinter**. It automates attendance marking using facial recognition, making it ideal for **educational institutions** and **workplaces**.

## Features
-  **Face Registration**: Capture and store multiple images per student.
-  **Face Recognition**: Identify students in real-time and mark attendance.
-  **Database Management**: Store student details and attendance records in SQLite.
-  **Attendance Reports**: Export **daily/monthly** attendance in **Excel/CSV** format.
-  **GUI Interface**: Built with **Tkinter** for easy management.
-  **Data Organization**: Automatically organizes images and records.

## Technologies Used
- **Python** (Tkinter, OpenCV, NumPy, Pandas, SQLite)
- **Computer Vision** (Face Detection & Recognition using OpenCV)
- **Database Management** (SQLite)
- **Graphical User Interface** (Tkinter)

## How It Works
1. **Register a new student**: Capture multiple face images for better recognition.
2. **Train the system**: LBPH recognizer trains on collected images.
3. **Take attendance**: Recognizes faces from a live camera feed and marks attendance.
4. **Generate reports**: Export attendance records in Excel or CSV format.

---

## Installation & Usage
### Prerequisites
- Python 3.x
- Required libraries: `tkinter`, `opencv-python`, `numpy`, `pandas`, `sqlite3`, `tkcalendar`

### Steps
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/face-recognition-attendance.git
   cd face-recognition-attendance
   ```
2. Install dependencies:
   ```sh
   pip install opencv-python numpy pandas tkcalendar
   ```
3. Run the program:
   ```sh
   python mainnn.py
   ```

---

## Functionalities Used

### 1️ Graphical User Interface (GUI)
**Library Used: Tkinter**
- Uses `Tkinter` for buttons, labels, and tables.
- `Treeview` is used to display attendance records.

 **Key Functions:**
- `setup_gui()`: Initializes the main window.
- `register_new_person()`: Opens a student registration window.
- `manage_database()`: Displays and manages registered students.

### 2️ Face Detection & Recognition
**Library Used: OpenCV (cv2)**
- Face detection via **Haarcascade**.
- Face recognition via **LBPH algorithm**.

 **Key Functions:**
- `capture_face(name)`: Captures and stores face images.
- `train_recognizer()`: Trains an LBPH recognizer.
- `run_attendance()`: Recognizes faces in real-time and marks attendance.

### 3️ Student & Attendance Database
**Library Used: SQLite3**
- Stores student details and attendance logs.

 **Key Functions:**
- `create_student_tables()`: Creates student tables.
- `create_attendance_tables()`: Creates attendance tables.
- `mark_attendance(name)`: Inserts attendance into the database.

### 4️ Attendance Reports Generation
**Library Used: Pandas**
- Exports **daily/monthly** reports.

 **Key Functions:**
- `generate_daily_report()`: Exports daily reports.
- `generate_monthly_report()`: Exports monthly reports.

### 5️ File & Folder Management
**Libraries Used: OS, Shutil**
- Stores face images in `./student_database/photos/`.
- Saves reports in `./Reports/`.

 **Key Functions:**
- `delete_person(selection)`: Deletes a student’s images and records.
- `update_person_photo(selection)`: Captures new images for an existing student.

### 6️ Date Selection & Filtering
**Library Used: Tkcalendar**
- Filters attendance records by date.

 **Key Functions:**
- `show_calendar()`: Opens a date selection popup.
- `refresh_attendance_table()`: Filters attendance by selected date.

### 7️ Application Lifecycle & Cleanup
**Library Used: Datetime, OS**
- Ensures proper cleanup on exit.

 **Key Functions:**
- `__del__()`: Closes the database connection on exit.

---

## Future Enhancements 
- **Deep Learning Integration** (CNN) for better recognition.
- **Web-Based Interface** for remote access.
- **Cloud Database** for secure storage.

---

##  Contributing
Feel free to contribute by forking the repo and making a pull request!

---

###  **Let me know if you need any modifications!** 

