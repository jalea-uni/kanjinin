import express from "express";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import { KanjiPDFGenerator } from "./helpers/kanji-pdf-generator.mjs";

dotenv.config();

// Per ottenere __dirname in ES6 modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Funzione principale
 */
function main() {
  const port = process.argv[2] || process.env.PORT || 2990;

  try {
    console.log("🚀 Avvio Kanji Generator Server...\n");

    // Verifica che il font esista
    const fontPath = path.join(__dirname, process.env.FONT);
    if (!fs.existsSync(fontPath)) {
      throw new Error(`Font non trovato: ${fontPath}`);
    }

    startWebService(parseInt(port));
  } catch (error) {
    console.error("❌ Errore nell'avvio del server:", error.message);
    console.log("\n📋 Setup richiesto:");
    console.log("1. sudo apt-get install wkhtmltopdf");
    console.log("2. npm install express dotenv");
    console.log("3. Crea file .env con le variabili necessarie");
    console.log("4. Aggiungi font giapponese");
    process.exit(1);
  }
}

/**
 * Avvia il servizio web
 * @param {number} port - Porta del servizio
 */
function startWebService(port = 3000) {
  const app = express();

  app.use(express.json({ limit: "10mb" }));
  app.use(express.text({ limit: "10mb" }));

  app.post("/generate", async (req, res) => {
    try {
      const { content, options = {} } = req.body;

      if (!content) {
        return res.status(400).json({
          error: "Missing content field in request body",
        });
      }

      console.log(`🌐 Richiesta ricevuta - Lunghezza contenuto: ${content.length} caratteri`);

      // Verifica che il font esista
      const fontPath = path.join(__dirname, process.env.FONT);
      if (!fs.existsSync(fontPath)) {
        throw new Error(`Font non trovato: ${fontPath}`);
      }

      // Inizializza il generatore
      const generator = new KanjiPDFGenerator({
        outputDir: process.env.DATADIR || "temp_output",
        fontPath: fontPath,
      });

      // Processa il contenuto
      const result = await generator.processContent(content, {
        outputPdfName: options.outputPdfName || "kanji_sheet.pdf",
        outputJsonName: options.outputJsonName || "kanji_list.json",
        printJson: false,
      });

      // Risposta JSON
      const response = {
        success: true,
        data: {
          kanjiData: result.kanjiData,
          pdfBase64: result.pdfBase64,
          stats: result.stats,
        },
      };

      console.log(`✅ Risposta inviata - PDF: ${result.pdfBase64.length} caratteri base64`);
      res.json(response);
    } catch (error) {
      console.error("❌ Errore nel servizio web:", error.message);
      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  });

  // Endpoint di health check
  app.get("/health", (req, res) => {
    res.json({ status: "OK", timestamp: new Date().toISOString() });
  });

  // Endpoint per info sul servizio
  app.get("/info", (req, res) => {
    res.json({
      service: "Kanji Generator Server",
      version: "1.0.0",
      endpoints: {
        generate: "POST /generate",
        health: "GET /health",
        info: "GET /info",
      },
      example: {
        url: `http://localhost:${port}/generate`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: {
          content: "これは[例|れい]です。",
          options: {
            outputPdfName: "my_kanji.pdf",
            outputJsonName: "my_kanji.json",
          },
        },
      },
    });
  });

  app.listen(port, () => {
    console.log(`🌐 Servizio Kanji Generator attivo su porta ${port}`);
    console.log(`📡 Endpoint: POST http://localhost:${port}/generate`);
    console.log(`🏥 Health check: GET http://localhost:${port}/health`);
    console.log(`ℹ️  Info servizio: GET http://localhost:${port}/info`);
    console.log(`\n📋 Esempio richiesta:`);
    console.log(`curl -X POST http://localhost:${port}/generate \\`);
    console.log(`  -H "Content-Type: application/json" \\`);
    console.log(`  -d '{"content":"これは[例|れい]です。"}'`);
  });
}

// Esegui solo se chiamato direttamente
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
