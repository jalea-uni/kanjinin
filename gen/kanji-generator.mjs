import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import { Command } from "commander";
import { KanjiPDFGenerator } from "./helpers/kanjiPdfGenerator.mjs";

dotenv.config();

// Per ottenere __dirname in ES6 modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Funzione principale per CLI
 */
async function main() {
  const program = new Command();

  program
    .name("kanji-generator")
    .description("Genera PDF di pratica kanji da file .kni")
    .version("1.0.0")
    .option("--input <file>", "File .kni di input", process.env.INPUT || "input.kni")
    .option("--outputdir <dir>", "Directory di output", process.env.DATADIR || "export_data")
    .option("--outputpdf <file>", "Nome del PDF di output", process.env.PDF || "kanji_sheet.pdf")
    .option(
      "--outputjson <file>",
      "Nome del JSON di output",
      process.env.DATAFILE || "kanji_list.json"
    )
    .option("--print", "Stampa il JSON dei kanji", false);

  program.parse();
  const options = program.opts();

  const inputFile = path.join(__dirname, options.input);

  try {
    console.log("🚀 Avvio generazione PDF kanji...\n");

    // Verifica che il font esista
    const fontPath = path.join(__dirname, process.env.FONT);
    if (!fs.existsSync(fontPath)) {
      throw new Error(`Font non trovato: ${fontPath}`);
    }

    // Verifica che il file .kni esista
    if (!fs.existsSync(inputFile)) {
      throw new Error(`File .kni non trovato: ${inputFile}`);
    }

    // Leggi il contenuto del file
    const content = fs.readFileSync(inputFile, "utf8");

    // Inizializza il generatore
    const generator = new KanjiPDFGenerator({
      outputDir: options.outputdir,
      fontPath: fontPath,
    });

    // Processa il contenuto
    const result = await generator.processContent(content, {
      outputPdfName: options.outputpdf,
      outputJsonName: options.outputjson,
      printJson: options.print,
    });

    console.log("\n✨ Processo completato con successo!");
    console.log(`📊 Statistiche:`);
    console.log(`   - Righe processate: ${result.stats.linesProcessed}`);
    console.log(`   - Kanji unici: ${result.stats.uniqueKanji}`);
    console.log(`   - PDF: ${result.stats.pdfPath}`);
    console.log(`   - JSON: ${result.stats.jsonPath}`);
  } catch (error) {
    console.error("❌ Errore:", error.message);
    console.log("\n📋 Setup richiesto:");
    console.log("1. sudo apt-get install wkhtmltopdf");
    console.log("2. npm install dotenv commander");
    console.log("3. Crea file .env con le variabili necessarie");
    console.log("4. Aggiungi font giapponese e file .kni");
    process.exit(1);
  }
}

// Esegui solo se chiamato direttamente
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
