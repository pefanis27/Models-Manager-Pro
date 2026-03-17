# Models-Manager-Pro v6.0
The application Models Manager Pro v6.0 (A.I Copilot Edition), developed by Evangelos Pefanis, is an advanced and comprehensive environment for training, managing, and evaluating artificial intelligence models
. It is highly optimized for the YOLO (v5 through v12) architectures and CNN Classifiers from torchvision, providing full support for Python 3.12 and modern libraries like PyTorch, Ultralytics, and PySide6

Video demo of the new application under development in version 6.0 : https://drive.google.com/file/d/1OLzDfJ-37BzTueEmlOBkmEvD1N-RV-qM/view?usp=sharing

Download full installer: under development

![banner_v_6](banner_v_6.png)
The application Models Manager Pro v6.0 (A.I Copilot Edition), developed by Evangelos Pefanis, is an advanced and comprehensive environment for training, managing, and evaluating artificial intelligence models
. It is highly optimized for the YOLO (v5 through v12) architectures and CNN Classifiers from torchvision, providing full support for Python 3.12 and modern libraries like PyTorch, Ultralytics, and PySide6
.
Below is the detailed description of the application's capabilities by category:
1. Model Management and Training
The application covers the entire model life cycle, from data preparation to final reporting:
Supported Models: Includes all versions of YOLO (Detection & Classification) and popular CNN models such as MobileNet V2/V3 (Small/Large) and ResNet-50/101
.
Dataset Management: Features tools for the automatic download and preparation of well-known datasets like COCO (~20GB including images and labels), as well as support for custom ImageFolder structures
.
Full Customization: Users have absolute control over hyperparameters, including epochs, batch size, image size (ranging from 160 to 1920px), optimizers (SGD, Adam, AdamW), learning rate, momentum, weight decay, and the number of workers
.
Training Optimization: Integrates TorchInductor/Triton technology to accelerate training on NVIDIA GPUs through compilation, offering three modes: Default, Reduce-overhead, and Max-autotune
.
Automation: Includes an Early Stopping (patience) system to prevent overfitting and automatically saves both the "best" and "last" models
.
2. A.I Copilot (Artificial Intelligence)
Through the integration of the Groq API (LLM), the application offers an intelligent assistant that operates on three levels
:
Setting Suggestions: Proposes optimal initial parameters based on a user's description of their goal (e.g., "maximum accuracy" or "real-time speed")
.
Training Improvement: Analyzes logs and metrics from the last training run to suggest corrective actions
.
Detection Improvement: Utilizes detection statistics results to propose new training configurations
.
Immediate Application: The AI generates YAML configurations that can be automatically applied to the application's fields, marking changes with a 🤖 icon
.
3. Live Detection and Video Inference
Live Camera: Performs real-time detection and classification from a camera (targeting 1080p resolution) with support for multiple backends: PyTorch, ONNX, NCNN, and TensorRT (.engine)
.
Video Inference: Processes video files (mp4, avi, mov, etc.) with the ability to save the annotated result or individual frames
.
Specialized Overlay: For CNN models, it displays a dedicated panel showing the top-5 predictions with confidence bars and color coding, which scales automatically based on the frame resolution
.
Classes Filter: Provides the ability to filter which classes are displayed on the screen
.
4. Evaluation and Analytics
Detection Statistics: Conducts batch analysis of images (up to 500) to calculate class distribution, average confidence, and inference times per image, generating detailed PDF reports
.
Training Comparison: A specialized dialog for comparing runs that includes:
Metrics Table: 18 columns with performance-based cell coloring (green/amber/red)
.
Interactive Charts: 11 types of charts (Radar profile, F1-Score bars, Precision-Recall scatter, Loss curves, and Model Efficiency)
.
Best Model Highlighting: Automatically recommends a "winner" across 10 categories (e.g., Highest mAP50, Best Efficiency, Lowest Overfitting Index)
.
5. Export and Reporting
Model Export: Converts PyTorch models (.pt) into ONNX, TensorRT (for maximum speed on NVIDIA GPUs), and NCNN (for mobile/edge devices), while automatically creating metadata JSON files
.
Professional PDF Reports: Generates rich reports including cover pages, metrics tables, loss/accuracy curves, and detection samples, featuring improved typography and emoji-labeled section headers
.
6. User Interface and System Tools
Modern UI: A GitHub-inspired design with full support for Dark and Light themes (Arctic Cyber and Cyber Noir) and font zoom capabilities
.
Monitoring: Provides live tracking of CPU, RAM, and GPU VRAM consumption in the status bar
.
Diagnostics: A tool that collects complete system information and exports it to a ZIP file for troubleshooting
.
Safety and Stability: Features an automatic Crash Log system (with thread dumps), an integrated faulthandler for C-level errors, and smart GPU/CPU memory cleanup utilities
.
System Tray: An icon in the notification area for quick access and status updates
.

# Models-Manager-Pro v6.0 (Greek)
![Project_Manager_Pro_Ver_6_0](Project_Manager_Pro_Ver_6_0.png)
Η εφαρμογή Models Manager Pro v6.0 (A.I Copilot Edition), δημιουργία του Ευάγγελου Πεφάνη (2026), αποτελεί ένα προηγμένο και ολοκληρωμένο οικοσύστημα για τη διαχείριση του πλήρους κύκλου ζωής μοντέλων Τεχνητής Νοημοσύνης
. Σχεδιασμένη για να γεφυρώσει το χάσμα μεταξύ της έρευνας και της παραγωγικής εφαρμογής, η έκδοση 6.0 ενσωματώνει τεχνολογίες αιχμής για την εκπαίδευση, τη βελτιστοποίηση και την ανάπτυξη μοντέλων σε πραγματικές συνθήκες
.
Ακολουθεί η αναλυτική έκθεση των προδιαγραφών, των δυνατοτήτων και των υποσυστημάτων της εφαρμογής:
1. Τεχνική Στοίβα και Αρχιτεκτονική
Η εφαρμογή βασίζεται σε ένα σύγχρονο τεχνολογικό υπόβαθρο που εξασφαλίζει μέγιστη απόδοση και συμβατότητα:
Γλώσσα και Frameworks: Αναπτυγμένη σε Python 3.12 με γραφικό περιβάλλον PySide6 (Qt6)
.
Πυρήνας Μηχανικής Μάθησης: Χρησιμοποιεί τα frameworks PyTorch και torchvision για τη διαχείριση των νευρωνικών δικτύων
.
Υποστηριζόμενα Μοντέλα:
YOLO (Ultralytics): Πλήρης υποστήριξη όλων των εκδόσεων από v5 έως και την ολοκαίνουργια v12 για εργασίες ανίχνευσης (Detection) και ταξινόμησης (Classification)
.
CNN Ταξινομητές: Ενσωματωμένη υποστήριξη για αρχιτεκτονικές όπως MobileNet V2/V3 (Small/Large) και ResNet-50/101
.
Διαχείριση Νημάτων (Thread Isolation): Οι βαριές διεργασίες (εκπαίδευση, εξαγωγή, ανάλυση) εκτελούνται σε απομονωμένα νήματα, διατηρώντας τη διεπαφή χρήστη (GUI) 100% αποκρίσιμη
.
2. Υποσύστημα Εκπαίδευσης και Βελτιστοποίησης Υλικού
Το υποσύστημα αυτό επιτρέπει τον πλήρη έλεγχο της διαδικασίας δημιουργίας ενός μοντέλου:
Παραμετροποίηση: Έλεγχος όλων των κρίσιμων υπερπαραμέτρων, όπως batch size, epochs, image size, ρυθμός μάθησης (LR), optimizer (SGD, Adam, AdamW), momentum και weight decay
.
Τεχνολογία Triton / TorchInductor: Υποστηρίζει compile-time βελτιστοποίηση για μοντέλα YOLO σε κάρτες NVIDIA, προσφέροντας τρεις καταστάσεις λειτουργίας: Προεπιλογή, Μείωση επιβάρυνσης (reduce-overhead) και Μέγιστος αυτόματος συντονισμός (max-autotune)
.
Αυτόματη Διαχείριση Datasets: Σαρώνει αυτόματα τους φακέλους δεδομένων, υποστηρίζει πρότυπα όπως το COCO και διενεργεί ελέγχους εγκυρότητας στα αρχεία ετικετών (labels)
.
3. A.I Copilot: Έξυπνος Βοηθός Εκπαίδευσης
Πρόκειται για το πλέον καινοτόμο τμήμα της εφαρμογής, το οποίο λειτουργεί ως "προσωπικός MLOps Engineer":
Διασύνδεση LLM: Χρησιμοποιεί το Groq API για πρόσβαση σε Large Language Models (όπως Llama-3 ή Kimi)
.
Έξυπνες Προτάσεις: Αναλύει το hardware του χρήστη, τις τρέχουσες ρυθμίσεις και τα αποτελέσματα προηγούμενων εκπαιδεύσεων για να προτείνει βέλτιστες παραμέτρους
.
Αυτοματοποίηση: Δημιουργεί αυτόματα μπλοκ ρυθμίσεων (YAML) και τα εφαρμόζει απευθείας στα πεδία της φόρμας εκπαίδευσης με ένα κλικ
.
4. Υποσύστημα Εξαγωγής και Deployment
Η εφαρμογή προσφέρει ένα πλήρες pipeline μετατροπής των μοντέλων για χρήση σε διαφορετικές πλατφόρμες
:
ONNX (.onnx): Για γενική χρήση σε CPU και GPU μέσω του ONNX Runtime
.
TensorRT (.engine): Η απόλυτη επιλογή για μέγιστη ταχύτητα σε NVIDIA GPUs, με υποστήριξη FP16 precision και αυτόματο σύστημα διαχείρισης cache για αποφυγή επαναλαμβανόμενων χρονοβόρων builds
.
NCNN: Βελτιστοποιημένη εξαγωγή για mobile και embedded συσκευές (ARM NEON optimized), επιτρέποντας την εκτέλεση μοντέλων χωρίς εξαρτήσεις από Python ή PyTorch
.
5. Live Ανίχνευση και Video Inference
Real-Time Inference: Υποστηρίζει ζωντανή ροή από κάμερα με ταχύτητες έως και 60 FPS, χρησιμοποιώντας βελτιστοποιημένα backends (DSHOW/MSMF) για Windows
.
Adaptive Overlay: Διαθέτει ένα εξελιγμένο οπτικό επίπεδο (overlay) με badges κλάσεων και μπάρες εμπιστοσύνης (confidence bars) που προσαρμόζονται αυτόματα στην ανάλυση της εικόνας
.
Επεξεργασία Βίντεο: Δυνατότητα frame-by-frame επεξεργασίας αρχείων βίντεο (mp4, avi, κ.λπ.) με αυτόματη αποθήκευση των αποτελεσμάτων
.
6. Αξιολόγηση, Στατιστική Ανάλυση και Benchmarking
Batch Analysis: Δυνατότητα ανάλυσης έως και 500 εικόνων ενός dataset για την εξαγωγή αναλυτικών μετρικών ανά κλάση (True Positives, False Positives, μέσο confidence)
.
Benchmarking Suite: Περιλαμβάνει στατικό benchmark (offline) για σύγκριση ταχύτητας μεταξύ backends και benchmark κάμερας (live) για μέτρηση FPS σε πραγματικές συνθήκες
.
Σύγκριση Εκπαιδεύσεων: Ένα ενιαίο παράθυρο σύγκρισης όλων των αποθηκευμένων runs με 18 στήλες δεδομένων, 7 διαδραστικά γραφήματα και αυτόματη σύσταση του "Καλύτερου Μοντέλου" βάσει 7 διαφορετικών κριτηρίων (mAP, F1-Score, Efficiency κ.ά.)
.
7. Σύστημα Αναφορών και Σταθερότητας
PDF Reports: Αυτόματη παραγωγή επαγγελματικών αναφορών εκπαίδευσης και ανίχνευσης μέσω της βιβλιοθήκης ReportLab, που περιλαμβάνουν πίνακες μετρικών, καμπύλες loss/accuracy και πληροφορίες hardware
.
Διαχείριση Σφαλμάτων: Διαθέτει προηγμένο σύστημα καταγραφής Crash Logs με πλήρες traceback και thread dump, καθώς και διαγνωστικά εργαλεία για την κατάσταση της GPU και της VRAM
.
Ασφάλεια Υλικού (Hardware Guards): Ενσωματώνει μηχανισμούς κλειδώματος της κάμερας (camera lock) για την αποτροπή συγκρούσεων μεταξύ διαφορετικών tabs και έξυπνη εκκαθάριση μνήμης GPU/CPU
.
8. Σχεδιασμός και Εμπειρία Χρήστη (UI/UX)
Η έκδοση 6.0 προσφέρει ένα πλήρως ανανεωμένο περιβάλλον με:
Θέματα Dark & Light: Χρωματικές παλέτες εμπνευσμένες από το GitHub για υψηλή αναγνωσιμότητα
.
Splash Screen: Δυναμική οθόνη εκκίνησης με animations και gradients
.
Προσβασιμότητα: Εργαλεία ζουμ γραμματοσειράς, πλήρη υποστήριξη συντομεύσεων πληκτρολογίου και live dashboard παρακολούθησης πόρων συστήματος

7. Λήψη πλήρους προγράμματος εγκατάστασης: υπό ανάπτυξη
