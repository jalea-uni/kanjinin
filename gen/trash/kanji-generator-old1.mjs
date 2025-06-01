import fs from "fs";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
dotenv.config();

const execAsync = promisify(exec);

// Per ottenere __dirname in ES6 modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class KanjiPDFGenerator {
  constructor() {
    this.kanjiOutputDir = process.env.DATADIR;

    // Assicurati che la directory per i kanji esista
    if (!fs.existsSync(this.kanjiOutputDir)) {
      fs.mkdirSync(this.kanjiOutputDir, { recursive: true });
    }
  }

  /**
   * Legge e parsa il file .kni
   * @param {string} filePath - Percorso del file .kni
   * @returns {Array} Array di righe parsate
   */
  parseKniFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, "utf8");
      const lines = content.split("\n").filter((line) => line.trim() !== "");

      return lines.map((line) => this.parseLine(line));
    } catch (error) {
      console.error(`Errore nella lettura del file ${filePath}:`, error.message);
      throw error;
    }
  }

  /**
   * Parsa una singola riga del formato [kanji|furigana]
   * @param {string} line - Riga da parsare
   * @returns {Array} Array di oggetti {type, char, furigana}
   */
  parseLine(line) {
    const pattern = /\[([^|]+)\|([^\]]+)\]/g;
    const result = [];
    let lastIndex = 0;
    let match;

    console.log(`DEBUG: Parsing line: "${line}"`);

    while ((match = pattern.exec(line)) !== null) {
      // Aggiungi caratteri normali prima del match
      if (match.index > lastIndex) {
        const normalText = line.substring(lastIndex, match.index);
        for (const char of normalText) {
          if (char.trim()) {
            result.push({
              type: "normal",
              char: char,
              furigana: null,
            });
          }
        }
      }

      // Aggiungi kanji con furigana
      const kanji = match[1];
      const furigana = match[2];

      console.log(
        `DEBUG: Found match - Kanji: "${kanji}" (${kanji.length} chars), Furigana: "${furigana}" (${furigana.length} chars)`
      );

      // Gestisci kanji multipli con distribuzione corretta del furigana
      if (kanji.length === 1) {
        // Kanji singolo - tutto il furigana
        result.push({
          type: "kanji",
          char: kanji,
          furigana: furigana,
        });
        console.log(`DEBUG:   -> Single kanji: ${kanji} = ${furigana}`);
      } else {
        // Kanji multipli - distribuzione intelligente
        const furiganaPerKanji = Math.ceil(furigana.length / kanji.length);
        for (let i = 0; i < kanji.length; i++) {
          const startIdx = i * furiganaPerKanji;
          const endIdx = Math.min(startIdx + furiganaPerKanji, furigana.length);
          const kanjiChar = kanji[i];
          const kanjiReading = furigana.substring(startIdx, endIdx);

          result.push({
            type: "kanji",
            char: kanjiChar,
            furigana: kanjiReading,
          });
          console.log(
            `DEBUG:   -> Multi kanji [${i}]: ${kanjiChar} = ${kanjiReading} (chars ${startIdx}-${endIdx})`
          );
        }
      }

      lastIndex = pattern.lastIndex;
    }

    // Aggiungi caratteri rimanenti
    if (lastIndex < line.length) {
      const remaining = line.substring(lastIndex);
      for (const char of remaining) {
        if (char.trim()) {
          result.push({
            type: "normal",
            char: char,
            furigana: null,
          });
        }
      }
    }

    console.log(`DEBUG: Final result for line: ${result.length} items`);
    return result;
  }

  /**
   * Salva tutti i kanji in un singolo file JSON
   * @param {Array} parsedLines - Righe parsate
   */
  saveKanjiToFiles(parsedLines) {
    const kanjiSet = new Set();
    const kanjiArray = [];

    // Raccogli tutti i kanji unici
    parsedLines.forEach((line) => {
      line.forEach((item) => {
        if (item.type === "kanji") {
          if (!kanjiSet.has(item.char)) {
            kanjiSet.add(item.char);
            kanjiArray.push({
              kanji: item.char,
              furigana: item.furigana,
              unicode: `U+${item.char.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")}`,
            });
          }
        }
      });
    });

    // Salva tutto in un file JSON
    const jsonFilePath = path.join(this.kanjiOutputDir, process.env.DATAFILE);
    const jsonContent = JSON.stringify(kanjiArray, null, 2);

    try {
      fs.writeFileSync(jsonFilePath, jsonContent, "utf8");
      console.log(`📁 Salvati ${kanjiArray.length} kanji unici in ${jsonFilePath}`);
    } catch (error) {
      console.error(`Errore nel salvare il file JSON:`, error.message);
      throw error;
    }

    return kanjiArray;
  }

  /**
   * Genera l'HTML con CSS per il PDF
   * @param {Array} parsedLines - Righe parsate
   * @returns {string} HTML completo
   */
  generateHTML(parsedLines) {
    const fontPath = path.join(__dirname, process.env.FONT);
    const fontBase64 = fs.readFileSync(fontPath).toString("base64");

    // Parametri di layout da .env con valori di default
    const normalFontSize = process.env.NORMAL_FONT_SIZE ? process.env.NORMAL_FONT_SIZE : "16";
    const furiganaFontSize = process.env.FURIGANA_FONT_SIZE ? process.env.FURIGANA_FONT_SIZE : "10";
    const charSpacing = process.env.CHAR_SPACING ? process.env.CHAR_SPACING : "2";
    const kanjiSpacing = process.env.KANJI_SPACING ? process.env.KANJI_SPACING : "20";
    const boxSize = process.env.BOX_SIZE ? process.env.BOX_SIZE : "40";
    const lineSpacing = process.env.LINE_SPACING ? process.env.LINE_SPACING : "40";

    console.log(`DEBUG: Layout parameters from .env:`);
    console.log(`  - NORMAL_FONT_SIZE from env: "${process.env.NORMAL_FONT_SIZE}"`);
    console.log(`  - FURIGANA_FONT_SIZE from env: "${process.env.FURIGANA_FONT_SIZE}"`);
    console.log(`  - CHAR_SPACING from env: "${process.env.CHAR_SPACING}"`);
    console.log(`  - KANJI_SPACING from env: "${process.env.KANJI_SPACING}"`);
    console.log(`  - BOX_SIZE from env: "${process.env.BOX_SIZE}"`);
    console.log(`  - LINE_SPACING from env: "${process.env.LINE_SPACING}"`);

    console.log(`DEBUG: Final layout parameters:`);
    console.log(`  - Normal font size: ${normalFontSize}px`);
    console.log(`  - Furigana font size: ${furiganaFontSize}px`);
    console.log(`  - Character spacing: ${charSpacing}px`);
    console.log(`  - Kanji spacing: ${kanjiSpacing}px`);
    console.log(`  - Box size: ${boxSize}px`);
    console.log(`  - Line spacing: ${lineSpacing}px`);

    let html = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @font-face {
            font-family: 'Japanese';
            src: url(data:font/truetype;base64,${fontBase64}) format('truetype');
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Japanese', sans-serif;
            font-size: ${normalFontSize}px;
            line-height: 1;
            padding: 30px;
            background: white;
        }
        
        .page {
            width: 210mm;
            min-height: 297mm;
            margin: 0 auto;
            background: white;
        }
        
        .sentence {
            display: flex;
            flex-wrap: wrap;
            align-items: flex-end;
            margin-bottom: ${lineSpacing}px;
            gap: ${charSpacing}px; /* Spazio tra caratteri normali */
        }
        
        .normal-char {
            font-size: ${normalFontSize}px;
            line-height: 1;
            padding: 2px;
            display: inline-block;
        }
        
        .kanji-container {
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            margin: 0 ${kanjiSpacing}px; /* Spazio intorno ai kanji */
            position: relative;
        }
        
        .furigana {
            font-size: ${furiganaFontSize}px;
            line-height: 1;
            margin-bottom: 5px;
            text-align: center;
            white-space: nowrap;
        }
        
        .kanji-box {
            width: ${boxSize}px;
            height: ${boxSize}px;
            border: 2px solid #333;
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
        }
        
        /* Per la stampa */
        @media print {
            body { margin: 0; padding: 20px; }
            .page { margin: 0; }
        }
        
        @page {
            size: A4;
            margin: 20mm;
        }
    </style>
</head>
<body>
    <div class="page">`;

    // Genera HTML per ogni riga
    parsedLines.forEach((line, lineIndex) => {
      html += `\n        <div class="sentence">`;

      line.forEach((item, itemIndex) => {
        if (item.type === "kanji") {
          html += `
            <div class="kanji-container">
                <div class="furigana">${item.furigana || ""}</div>
                <div class="kanji-box"></div>
            </div>`;
        } else if (item.type === "normal") {
          html += `<span class="normal-char">${item.char}</span>`;
        }
      });

      html += `\n        </div>`;
    });

    html += `
    </div>
</body>
</html>`;

    return html;
  }

  /**
   * Crea il PDF usando wkhtmltopdf
   * @param {Array} parsedLines - Righe parsate
   * @param {string} outputPath - Percorso del PDF di output
   */
  async createPDF(parsedLines, outputPath) {
    console.log(`DEBUG: Generating HTML for ${parsedLines.length} lines`);

    // Genera l'HTML
    const html = this.generateHTML(parsedLines);

    // Salva l'HTML temporaneo
    const htmlPath = path.join(this.kanjiOutputDir, "temp.html");
    fs.writeFileSync(htmlPath, html, "utf8");
    console.log(`DEBUG: HTML temporaneo salvato in ${htmlPath}`);

    // Salva anche per debug
    const debugHtmlPath = path.join(this.kanjiOutputDir, "debug.html");
    fs.writeFileSync(debugHtmlPath, html, "utf8");
    console.log(`DEBUG: HTML debug salvato in ${debugHtmlPath}`);

    try {
      // Comando wkhtmltopdf
      const command = `wkhtmltopdf --page-size A4 --margin-top 20mm --margin-right 20mm --margin-bottom 20mm --margin-left 20mm --encoding UTF-8 "${htmlPath}" "${outputPath}"`;

      console.log(`DEBUG: Eseguendo comando: ${command}`);

      const { stdout, stderr } = await execAsync(command);

      if (stderr && !stderr.includes("Warning")) {
        console.warn(`wkhtmltopdf warnings: ${stderr}`);
      }

      // Rimuovi file temporaneo
      fs.unlinkSync(htmlPath);

      console.log(`📄 PDF creato: ${outputPath}`);
    } catch (error) {
      console.error(`Errore wkhtmltopdf: ${error.message}`);
      throw error;
    }
  }
}

/**
 * Funzione principale
 */
async function main() {
  const inputFile = path.join(__dirname, process.env.INPUT);
  const outputPdf = path.join(__dirname, process.env.DATADIR, process.env.PDF);

  try {
    console.log("🚀 Avvio generazione PDF kanji con HTML + wkhtmltopdf...\n");

    // Verifica che il font esista
    const fontPath = path.join(__dirname, process.env.FONT);
    if (!fs.existsSync(fontPath)) {
      throw new Error(`Font non trovato: ${fontPath}`);
    }

    // Verifica che il file .kni esista
    if (!fs.existsSync(inputFile)) {
      throw new Error(`File .kni non trovato: ${inputFile}`);
    }

    // Inizializza il generatore
    const generator = new KanjiPDFGenerator();

    // Parsa il file .kni
    console.log(`📖 Lettura file: ${inputFile}`);
    const parsedLines = generator.parseKniFile(inputFile);

    console.log(`✅ Parsate ${parsedLines.length} righe`);

    // Debug: mostra il parsing
    parsedLines.forEach((line, index) => {
      console.log(
        `Riga ${index + 1}:`,
        line
          .map((item) => (item.type === "kanji" ? `${item.char}(${item.furigana})` : item.char))
          .join(" ")
      );
    });

    // Salva i kanji in file JSON
    console.log("\n💾 Salvataggio kanji...");
    const kanjiData = generator.saveKanjiToFiles(parsedLines);

    // Crea il PDF
    console.log("\n🎨 Creazione PDF con HTML + wkhtmltopdf...");
    await generator.createPDF(parsedLines, outputPdf);

    console.log("\n✨ Processo completato con successo!");
    console.log(`📊 Statistiche:`);
    console.log(`   - Righe processate: ${parsedLines.length}`);
    console.log(`   - Kanji unici: ${kanjiData.length}`);
    console.log(`   - PDF: ${path.basename(outputPdf)}`);
    console.log(`   - JSON kanji: ${process.env.DATAFILE}`);
    console.log(`   - HTML debug: debug.html`);
  } catch (error) {
    console.error("❌ Errore:", error.message);
    console.log("\n📋 Setup richiesto:");
    console.log("1. sudo apt-get install wkhtmltopdf");
    console.log("2. npm install dotenv");
    console.log("3. Crea file .env con le variabili necessarie");
    console.log("4. Aggiungi font giapponese e file .kni");
  }
}

// Esegui solo se chiamato direttamente
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { KanjiPDFGenerator };
