const express = require('express');
const multer = require('multer');
const { execFile } = require('child_process');
const path = require('path');
const cors = require('cors');
const app = express();
const port = 3000;

app.use(cors({
  origin:"http://localhost:8080",
  credentials:true,
}));
// Configure multer
const upload = multer({ dest: 'uploads/' });

app.post('/evaluate', upload.fields([
  { name: 'modelFile', maxCount: 1 },
  { name: 'csvFile', maxCount: 1 }
]), (req, res) => {
  const modelPath = req.files?.modelFile?.[0]?.path;
  const csvPath = req.files?.csvFile?.[0]?.path;
  const modelType = req.body.modelType || "QAUM";

  if (!modelPath || !csvPath) {
    return res.status(400).json({ error: "Both model and CSV files are required." });
  }

  const scriptPath = path.join(__dirname, 'evaluate.py');

  execFile('python', [scriptPath, modelPath, csvPath, modelType], (error, stdout, stderr) => {
    if (error) {
      console.error('Python error:', { error, stderr });
      return res.status(500).json({ 
        error: 'Python script error',
        stderr: stderr.toString()
      });
    }

    try {
      const result = JSON.parse(stdout);
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: 'Invalid JSON output', raw: stdout });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});