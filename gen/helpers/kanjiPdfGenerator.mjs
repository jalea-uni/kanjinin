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
  constructor(options = {}) {
    // Usa opzioni passate o fallback a process.env
    this.outputDir = options.outputDir || process.env.DATADIR;
    this.fontPath = options.fontPath || path.join(path.dirname(__dirname), process.env.FONT);

    // Assicurati che la directory di output esista
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
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
   * @param {string} outputJsonName - Nome del file JSON di output
   */
  saveKanjiToFiles(parsedLines, outputJsonName) {
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
    const jsonFilePath = path.join(this.outputDir, outputJsonName);
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
   * Genera l'HTML con CSS per il PDF caricando il template da file
   * @param {Array} parsedLines - Righe parsate
   * @returns {string} HTML completo
   */
  generateHTML(parsedLines) {
    const fontBase64 = fs.readFileSync(this.fontPath).toString("base64");

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

    // Carica il template HTML da file
    const templatePath = path.join(path.dirname(__dirname), "template", "template.html");
    let html;

    try {
      html = fs.readFileSync(templatePath, "utf8");
      console.log(`DEBUG: Template caricato da ${templatePath}`);
    } catch (error) {
      console.error(`Template non trovato: ${templatePath}`);
      throw new Error(`Crea il file template in: ${templatePath}`);
    }

    // Sostituisci le variabili nel template
    html = html.replace("{{FONT_BASE64}}", fontBase64);
    html = html.replace(/\{\{\s*NORMAL_FONT_SIZE\s*\}\}/g, normalFontSize);
    html = html.replace(/\{\{\s*FURIGANA_FONT_SIZE\s*\}\}/g, furiganaFontSize);
    html = html.replace(/\{\{\s*CHAR_SPACING\s*\}\}/g, charSpacing);
    html = html.replace(/\{\{\s*KANJI_SPACING\s*\}\}/g, kanjiSpacing);
    html = html.replace(/\{\{\s*BOX_SIZE\s*\}\}/g, boxSize);
    html = html.replace(/\{\{\s*LINE_SPACING\s*\}\}/g, lineSpacing);

    // Genera HTML per ogni riga (CENTRATO VERTICALMENTE)
    let contentHTML = "";
    parsedLines.forEach((line, lineIndex) => {
      contentHTML += `\n        <div class="sentence">`;

      line.forEach((item, itemIndex) => {
        if (item.type === "kanji") {
          contentHTML += `
            <div class="kanji-container">
                <div class="furigana">${item.furigana || ""}</div>
                <div class="kanji-box"></div>
            </div>`;
        } else if (item.type === "normal") {
          // AGGIUNTO: Wrapper per centrare verticalmente i caratteri normali
          contentHTML += `
            <div class="normal-char-wrapper">
                <span class="normal-char">${item.char}</span>
            </div>`;
        }
      });

      contentHTML += `\n        </div>`;
    });

    // Sostituisci il contenuto nel template
    html = html.replace("{{CONTENT}}", contentHTML);

    return html;
  }

  /**
   * Crea il PDF usando wkhtmltopdf
   * @param {Array} parsedLines - Righe parsate
   * @param {string} outputPath - Percorso del PDF di output
   * @returns {Buffer} Buffer del PDF
   */
  async createPDF(parsedLines, outputPath) {
    console.log(`DEBUG: Generating HTML for ${parsedLines.length} lines`);

    // Genera l'HTML
    const html = this.generateHTML(parsedLines);

    // Salva l'HTML temporaneo
    const htmlPath = path.join(this.outputDir, "temp.html");
    fs.writeFileSync(htmlPath, html, "utf8");
    console.log(`DEBUG: HTML temporaneo salvato in ${htmlPath}`);

    // Salva anche per debug
    const debugHtmlPath = path.join(this.outputDir, "debug.html");
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

      // Leggi il PDF generato
      const pdfBuffer = fs.readFileSync(outputPath);

      // Rimuovi file temporaneo
      fs.unlinkSync(htmlPath);

      console.log(`📄 PDF creato: ${outputPath}`);

      return pdfBuffer;
    } catch (error) {
      console.error(`Errore wkhtmltopdf: ${error.message}`);
      throw error;
    }
  }

  /**
   * Processa il contenuto del file .kni e genera PDF e JSON
   * @param {string} content - Contenuto del file .kni
   * @param {Object} options - Opzioni per il processing
   * @returns {Object} Risultato con PDF buffer e dati JSON
   */
  async processContent(content, options = {}) {
    const {
      outputPdfName = "kanji_sheet.pdf",
      outputJsonName = "kanji_list.json",
      printJson = false,
    } = options;

    // Parsa il contenuto
    const lines = content.split("\n").filter((line) => line.trim() !== "");
    const parsedLines = lines.map((line) => this.parseLine(line));

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
    const kanjiData = this.saveKanjiToFiles(parsedLines, outputJsonName);

    // Stampa JSON se richiesto
    if (printJson) {
      console.log("\n📄 JSON dei kanji:");
      console.log(JSON.stringify(kanjiData, null, 2));
    }

    // Crea il PDF
    console.log("\n🎨 Creazione PDF...");
    const outputPdfPath = path.join(this.outputDir, outputPdfName);
    const pdfBuffer = await this.createPDF(parsedLines, outputPdfPath);

    return {
      kanjiData,
      pdfBuffer,
      pdfBase64: pdfBuffer.toString("base64"),
      stats: {
        linesProcessed: parsedLines.length,
        uniqueKanji: kanjiData.length,
        pdfPath: outputPdfPath,
        jsonPath: path.join(this.outputDir, outputJsonName),
      },
    };
  }
}

export { KanjiPDFGenerator };
