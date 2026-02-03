// ============================================================
// Fiji / ImageJ Macro (.ijm)
// 3-channel TIFF -> Enrichment Ratio (I_Receptor / I_Total)
// - Background correction (rolling ball) on receptor & CPP channels
// - Whole cell mask: CPP channel threshold (Li) + Analyze Particles + Combine
// - Receptor mask: receptor channel threshold (Yen) + Analyze Particles + Combine
// - Measure mean intensity on background-corrected CPP image
// Output CSV: filename, I_Total, I_Receptor, Ratio, WholeCellArea, ReceptorArea, Flags
// ============================================================

// --------------------
// User parameters
// --------------------
// After Split Channels, channels are usually C1, C2, C3.
// Set which channel is receptor (Red) and which is CPP (Green).
RECEPTOR_CH = 1;   // 1=C1, 2=C2, 3=C3
CPP_CH      = 2;   // 1=C1, 2=C2, 3=C3

ROLLING_BALL = 50;     // pixels
WHOLE_MIN_SIZE = 50;   // pixels^2
RECEPTOR_MIN_SIZE = 5; // pixels^2
RECEPTOR_CIRC_MIN = 0.00;
RECEPTOR_CIRC_MAX = 1.00;

DEBUG_SAVE_MASKS = true;   // trueで二値マスクも保存
DEBUG_PRINT_TITLES = true; // trueでチャンネル割当をLogに出す

// Threshold polarity: usually objects are bright on dark background => "dark"
WHOLE_THRESH  = "Li dark";
RECEPT_THRESH = "Shanbhag dark";

// Debug overlay export (optional)
SAVE_OVERLAY = true; // true にすると輪郭画像を保存
SAVE_SEPARATE_OVERLAYS = true; // 赤に受容体ROI、緑に細胞ROIを別々に保存

// --------------------
// Helper: select multiple ROI indices
// --------------------
function selectRoisByIndexArray(idxArr) {
    roiManager("Deselect");
    roiManager("Select", idxArr);
}

function stripExt(fname) {
    dot = lastIndexOf(fname, ".");
    if (dot == -1) return fname;
    return substring(fname, 0, dot);
}

// Helper: find channel window title robustly for both
//  - "C1-<orig>", "C2-<orig>", "C3-<orig>" (composite/hyperstack)
//  - "<orig> (red)", "<orig> (green)", "<orig> (blue)" (RGB split)
function findChannelTitle(chIndex, origTitle) {

    // 1) C1- style
    want = "C" + chIndex + "-" + origTitle;
    if (isOpen(want)) return want;

    titles = getList("image.titles");

    prefix = "C" + chIndex + "-";
    for (t=0; t<titles.length; t++) {
        if (startsWith(titles[t], prefix)) return titles[t];
    }

    // 2) RGB style suffix
    if (chIndex==1) suffix = " (red)";
    else if (chIndex==2) suffix = " (green)";
    else suffix = " (blue)";

    want2 = origTitle + suffix;
    if (isOpen(want2)) return want2;

    for (t=0; t<titles.length; t++) {
        if (endsWith(titles[t], suffix)) return titles[t];
    }

    return "";
}

// --------------------
// Main
// --------------------
requires("1.53");
setBatchMode(true);

// Measurements: mean + area
run("Set Measurements...", "mean area decimal=6");

inputDir = getDirectory("Choose folder containing tif/tiff files");
if (inputDir=="") exit("No folder selected.");

outDir = getDirectory("Choose folder to save CSV");
if (outDir=="") exit("No output folder selected.");
outPath = outDir + "enrichment_results.csv";

// Write header
File.saveString("filename,I_Total,I_Receptor,Ratio,WholeCellArea,ReceptorArea,Flags\n", outPath);

list = getFileList(inputDir);

for (i=0; i<list.length; i++) {

    name = list[i];
    lower = toLowerCase(name);
    if (!(endsWith(lower, ".tif") || endsWith(lower, ".tiff"))) continue;

    flags = "";
    fullpath = inputDir + name;

    // reset
    run("Close All");
    roiManager("Reset");
    run("Clear Results");

    open(fullpath);
    origTitle = getTitle();

    // Try to split channels
	run("Split Channels");
	
	// 必ず初期化（未定義参照を防ぐ）
	c1 = ""; c2 = ""; c3 = "";
	recTitle = ""; cppTitle = "";
	
	// チャンネルウィンドウ名を取得
	c1 = findChannelTitle(1, origTitle);
	c2 = findChannelTitle(2, origTitle);
	c3 = findChannelTitle(3, origTitle);
	
	// ここで初めてデバッグ出力（c1が未定義にならない）
	if (DEBUG_PRINT_TITLES) {
	    print("FILE=" + name);
	    print(" origTitle=" + origTitle);
	    print(" c1=" + c1);
	    print(" c2=" + c2);
	    print(" c3=" + c3);
	}
	
	// 3チャンネルが取れないならスキップ（ここでも止まらない）
	if (c1=="" || c2=="" || c3=="") {
	    flags = flags + "SKIP_NOT_3CH;";
	    File.append(name + ",NA,NA,NA,NA,NA," + flags + "\n", outPath);
	    continue;
	}
	
	// map receptor & cpp（ここから先でrecTitle/cppTitleが決まる）
	if (RECEPTOR_CH==1) recTitle = c1;
	else if (RECEPTOR_CH==2) recTitle = c2;
	else recTitle = c3;
	
	if (CPP_CH==1) cppTitle = c1;
	else if (CPP_CH==2) cppTitle = c2;
	else cppTitle = c3;
	
	if (DEBUG_PRINT_TITLES) {
	    print(" recTitle=" + recTitle);
	    print(" cppTitle=" + cppTitle);
	}

    // -------------------------
    // Phase 1: Background correction (no 'light' option)
    // -------------------------
    selectWindow(recTitle);
    run("Subtract Background...", "rolling=" + ROLLING_BALL);

    selectWindow(cppTitle);
    run("Subtract Background...", "rolling=" + ROLLING_BALL);

    // duplicate corrected CPP for measurement
    selectWindow(cppTitle);
    run("Duplicate...", "title=CPP_meas");

    // -------------------------
    // Phase 2: Whole cell mask from CPP (Li)
    // -------------------------
    selectWindow(cppTitle);
    run("Duplicate...", "title=CPP_for_mask");
    selectWindow("CPP_for_mask");
    if (DEBUG_SAVE_MASKS) {
    	saveAs("PNG", outDir + stripExt(name) + "_CPP_mask.png");
	}

	setAutoThreshold(WHOLE_THRESH);
	run("Convert to Mask");
	if (DEBUG_SAVE_MASKS) {
	    saveAs("PNG", outDir + stripExt(name) + "_CPP_mask.png");
	}
	run("Analyze Particles...", "size=" + WHOLE_MIN_SIZE + "-Infinity show=Nothing add");
    
    wholeCount = roiManager("count");
    hasWhole = (wholeCount > 0);

    if (!hasWhole) {
        flags = flags + "NO_WHOLE_CELL_ROI;";
    } else {
        idx = newArray(wholeCount);
        for (k=0; k<wholeCount; k++) idx[k] = k;
        selectRoisByIndexArray(idx);
        roiManager("Combine");
        roiManager("Add"); // combined appended
        keep = roiManager("count") - 1;

        // delete originals (0..keep-1)
        for (k=keep-1; k>=0; k--) {
            roiManager("Select", k);
            roiManager("Delete");
        }
        // now Whole mask is index 0
    }

    // -------------------------
    // Phase 3: Receptor mask from receptor channel (Yen)
    // -------------------------
    selectWindow(recTitle);
    run("Duplicate...", "title=REC_for_mask");
    selectWindow("REC_for_mask");
    if (DEBUG_SAVE_MASKS) {
    	saveAs("PNG", outDir + stripExt(name) + "_REC_mask.png");
	}

	setAutoThreshold(RECEPT_THRESH);
	run("Convert to Mask");
	if (DEBUG_SAVE_MASKS) {
	    saveAs("PNG", outDir + stripExt(name) + "_REC_mask.png");
	}
	run("Analyze Particles...", "size=" + RECEPTOR_MIN_SIZE + "-Infinity circularity=" + RECEPTOR_CIRC_MIN + "-" + RECEPTOR_CIRC_MAX + " show=Nothing add");

    totalRois = roiManager("count");
	if (hasWhole) recStart = 1;
	else recStart = 0;
    recCount = totalRois - recStart;
    hasRec = (recCount > 0);

    if (!hasRec) {
        flags = flags + "NO_RECEPTOR_ROI;";
    } else {
        idx2 = newArray(recCount);
        for (k=0; k<recCount; k++) idx2[k] = recStart + k;
        selectRoisByIndexArray(idx2);
        roiManager("Combine");
        roiManager("Add"); // combined appended
        keep2 = roiManager("count") - 1;

        // delete original receptor ROIs (recStart..keep2-1)
        for (k=keep2-1; k>=recStart; k--) {
            roiManager("Select", k);
            roiManager("Delete");
        }
        // After deletion, receptor combined ROI index:
        // if hasWhole: index 1, else: index 0
    }

    // -------------------------
    // Measure on CPP_meas
    // -------------------------
    I_Total = "NA"; I_Rec = "NA"; ratio = "NA";
    wholeArea = "NA"; recArea = "NA";

    selectWindow("CPP_meas");

    if (hasWhole) {
        roiManager("Select", 0);
        run("Measure");
        row = nResults - 1;
        I_Total = getResult("Mean", row);
        wholeArea = getResult("Area", row);
        if (I_Total <= 0) flags = flags + "ZERO_TOTAL;";
    }

    if (hasRec) {
		if (hasWhole) recIndex = 1;
		else recIndex = 0;
        roiManager("Select", recIndex);
        run("Measure");
        row = nResults - 1;
        I_Rec = getResult("Mean", row);
        recArea = getResult("Area", row);
    }

    if (I_Total!="NA" && I_Rec!="NA") {
        if (I_Total > 0) ratio = I_Rec / I_Total;
        else ratio = "NA";
    }

	// -------------------------
	// Save overlays (separate)
	// - Red image: receptor ROI drawn on recTitle
	// - Green image: whole-cell ROI drawn on CPP_meas
	// -------------------------
	if (SAVE_SEPARATE_OVERLAYS) {
	
	    // 1) Green (CPP_meas) + Whole-cell ROI
	    if (hasWhole) {
	        selectWindow("CPP_meas");
	        run("Duplicate...", "title=CPP_OVERLAY_TMP");
	        selectWindow("CPP_OVERLAY_TMP");
	        run("RGB Color");
	
	        roiManager("Select", 0);          // Whole ROI is always index 0 if exists
	        setColor(255, 255, 0);            // yellow
	        run("Draw", "slice");
	
	        outCpp = outDir + stripExt(name) + "_CPP_wholeROI.png";
	        saveAs("PNG", outCpp);
	        close();
	    }
	
	    // 2) Red (recTitle) + Receptor ROI
	    if (hasRec) {
	        // receptor ROI index depends on whether Whole exists
	        recIndex = 0;
	        if (hasWhole) recIndex = 1;
	
	        selectWindow(recTitle);           // background-corrected red channel
	        run("Duplicate...", "title=REC_OVERLAY_TMP");
	        selectWindow("REC_OVERLAY_TMP");
	        run("RGB Color");
	
	        roiManager("Select", recIndex);
	        setColor(255, 255, 0);            // yellow
	        run("Draw", "slice");
	
	        outRec = outDir + stripExt(name) + "_REC_receptorROI.png";
	        saveAs("PNG", outRec);
	        close();
	    }
	}
    // write CSV
    line = name + "," + I_Total + "," + I_Rec + "," + ratio + "," + wholeArea + "," + recArea + "," + flags + "\n";
    File.append(line, outPath);
}

setBatchMode(false);
print("Done. Results saved to: " + outPath);