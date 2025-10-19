---
layout: page
title: Email Ledger Service
description: AI-Powered Email-to-Financial-Entry Automation
img: assets/img/email-ledger-service.png
importance: 1
category: "AI Services"
---

An automated email processing system that transforms receipts and invoices into categorized financial entries using vision AI models. The platform achieves zero-touch expense tracking by automatically processing email attachments, classifying transactions, and creating ledger entries.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/email-ledger-service.png" title="Intelligent Document Processing System" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Automated pipeline integrating Gmail, AI classification, and financial data entry.
</div>

## Technical Stack
- **Gmail Integration**: OAuth2 authentication with google-auth-oauthlib and Gmail API v1
- **AI Classification**: Vertex AI Vision for document analysis and categorization
- **Document Processing**: pypdf2, pdfplumber, PIL for multi-format support
- **Infrastructure**: Google Cloud Run, Cloud SQL PostgreSQL, Cloud Storage

## Key Features
**Automated Email Processing**: Multi-tenant system with parallel email processing across multiple clients using thread-safe Gmail API access

**AI Document Classification**: Vertex AI Vision analyzes receipts and invoices to classify as expense or income transactions with confidence scoring

**Multi-Format Support**: Processes PDFs, images (JPG/PNG), Excel spreadsheets, and HTML email content with OCR for scanned documents

**Gmail Label Management**: Automatic "AI Processed" labels for deduplication and tracking, preventing redundant processing

**Smart File Conversion**: Automatic conversion of PDFs and multi-page documents to optimized formats for AI processing

## Performance Metrics
- 100+ emails/minute processing throughput
- 90% AI document categorization based on chart of accounts (COA) mapping accuracy
- 100% successful attachment download and processing
- 90% zero-touch automation rate