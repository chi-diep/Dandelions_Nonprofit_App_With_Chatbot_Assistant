import express from 'express';
import pkg from 'pg';
import dotenv from 'dotenv';
import cors from 'cors';
import sgMail from '@sendgrid/mail';
import { Parser } from 'json2csv';

dotenv.config();
const { Pool } = pkg;
const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());

// set some extra security headers
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});

// make sure we have a SendGrid key before starting
if (!process.env.SENDGRID_API_KEY) {
  console.error("No SENDGRID_API_KEY found in .env. Exiting...");
  process.exit(1);
}
sgMail.setApiKey(process.env.SENDGRID_API_KEY);
console.log('Loaded SendGrid Key:', process.env.SENDGRID_API_KEY.slice(0, 8));

// setup postgres connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});

let currentCodes = {}; // hold MFA codes here as { email: { code, expiresAt } }

// every minute, clean up any expired codes
setInterval(() => {
  const now = Date.now();
  for (const [email, record] of Object.entries(currentCodes)) {
    if (now > record.expiresAt) delete currentCodes[email];
  }
}, 60000);

// endpoint to send an MFA code
app.post('/api/send-code', async (req, res) => {
  const { email } = req.body;
  if (!email) return res.status(400).json({ error: "Email is required" });

  const code = Math.floor(100000 + Math.random() * 900000);
  const expiresAt = Date.now() + 10 * 60 * 1000;
  currentCodes[email] = { code, expiresAt };

  // build the HTML email
  const htmlContent = `
  <div style="max-width:600px;margin:auto;font-family:sans-serif;border-radius:10px;overflow:hidden;border:1px solid #e0e0e0;box-shadow:0 4px 20px rgba(0,0,0,0.05);">
    <div style="background:linear-gradient(135deg, #d946ef, #facc15);padding:25px;text-align:center;">
      <h1 style="color:#fff;font-size:26px;">Welcome to Dandelions 🌷</h1>
      <p style="color:#ecf0f1;font-size:14px;">Let’s keep your account secure & our garden blooming</p>
    </div>
    <div style="padding:30px;background:#fff;">
      <p style="font-size:16px;color:#333;">Hello friend,</p>
      <p style="font-size:16px;color:#333;">Use this code to continue. It's valid for <strong>10 minutes</strong>.</p>
      <div style="text-align:center;margin:30px 0;">
        <div style="display:inline-block;background:#f0f9f5;border:2px dashed #facc15;border-radius:8px;padding:18px 35px;font-size:32px;color:#d946ef;letter-spacing:4px;font-weight:bold;">
          ${code}
        </div>
      </div>
      <p style="font-size:14px;color:#555;">If this wasn’t you, just ignore this email. Your data stays safe 🌱.</p>
    </div>
    <div style="background:#f6f6f6;text-align:center;padding:18px;">
      <small style="color:#999;">&copy; ${new Date().getFullYear()} Dandelions. All rights reserved.</small>
    </div>
  </div>`;

  try {
    await sgMail.send({
      to: email,
      from: { email: 'info@dandelions.org', name: 'Dandelions Security Team' },
      subject: '🌷 Your Dandelions verification code',
      html: htmlContent,
      text: `Hi there,\n\nYour one-time Dandelions code is: ${code}\n\nIt expires in 10 minutes.\n\n— Dandelions Team`
    });
    console.log(`Sent code ${code} to ${email}`);
    res.json({ message: "Code sent" });
  } catch (err) {
    console.error('SendGrid error:', err.response?.body || err);
    res.status(500).json({ error: "Failed to send email" });
  }
});

// endpoint to verify MFA code
app.post('/api/verify-code', async (req, res) => {
  const { email, code } = req.body;
  if (!email || !code) return res.status(400).json({ error: "Email and code required" });

  const record = currentCodes[email];
  if (!record) return res.status(401).json({ error: "No code found or already used" });
  if (Date.now() > record.expiresAt) {
    delete currentCodes[email];
    return res.status(401).json({ error: "Code expired" });
  }

  if (parseInt(code) === record.code) {
    delete currentCodes[email];

    let role = "volunteer";
    try {
      const check = await pool.query(`SELECT role FROM volunteers WHERE email=$1`, [email]);
      if (check.rows.length > 0) role = check.rows[0].role || "volunteer";
    } catch (err) {
      console.error("DB role lookup failed:", err);
    }

    console.log(`Verified ${email} as ${role}`);
    return res.json({ verified: true, role });
  }

  res.status(401).json({ error: "Invalid code" });
});

// endpoint for volunteers to sign up
app.post('/api/signup', async (req, res) => {
  const { name, email, phone, title, date, hours } = req.body;
  console.log(`New signup: ${name} for ${title} on ${date}`);

  try {
    await pool.query(
      `INSERT INTO signups (name, email, phone, shift_title, shift_date, shift_hours)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [name, email, phone, title, date, hours]
    );

    const htmlContent = `
    <div style="max-width:600px;margin:auto;font-family:sans-serif;border-radius:10px;overflow:hidden;border:1px solid #e0e0e0;box-shadow:0 4px 20px rgba(0,0,0,0.05);">
      <div style="background:linear-gradient(135deg,#d946ef,#facc15);padding:25px;text-align:center;">
        <h2 style="color:#fff;font-size:24px;">Thanks for signing up, ${name}!</h2>
        <p style="color:#ecf0f1;font-size:14px;">We're excited to see you at Dandelions 🌼</p>
      </div>
      <div style="padding:30px;background:#fff;">
        <p style="font-size:16px;color:#333;">You’re confirmed for:</p>
        <ul style="font-size:16px;color:#555;">
          <li><strong>Shift:</strong> ${title}</li>
          <li><strong>Date:</strong> ${date}</li>
          <li><strong>Hours:</strong> ${hours}</li>
        </ul>
        <p style="margin-top:20px;font-size:14px;color:#555;">Reply anytime if you have questions. We can’t wait to make memories together.</p>
      </div>
      <div style="background:#f6f6f6;text-align:center;padding:18px;">
        <small style="color:#999;">&copy; ${new Date().getFullYear()} Dandelions. All rights reserved.</small>
      </div>
    </div>`;

    await sgMail.send({
      to: email,
      from: { email: 'cheeyippi@gmail.com', name: 'Dandelions Team' },
      subject: `You're signed up for "${title}" on ${date}`,
      html: htmlContent,
      text: `Hi ${name},\n\nYou’re signed up for "${title}" on ${date} for ${hours} hours.\n\nSee you soon!\n\n— Dandelions Team`
    });

    res.json({ success: true });
  } catch (err) {
    console.error('Signup DB or SendGrid error:', err);
    res.status(500).json({ error: 'Could not save signup' });
  }
});

// endpoint to get signups or download as CSV for managers
app.get('/api/signups', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM signups ORDER BY created_at DESC');
    res.json(result.rows);
  } catch (err) {
    console.error('DB error fetching signups:', err);
    res.status(500).json({ error: 'Could not fetch signups' });
  }
});

app.get('/api/signups.csv', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM signups ORDER BY created_at DESC');
    const parser = new Parser({ fields: ['id', 'name', 'email', 'phone', 'shift_title', 'shift_date', 'shift_hours', 'created_at'] });
    const csv = parser.parse(result.rows);

    res.header('Content-Type', 'text/csv');
    res.attachment('dandelions_signups.csv');
    return res.send(csv);
  } catch (err) {
    console.error('CSV export failed:', err);
    res.status(500).json({ error: 'Could not generate CSV' });
  }
});

// simple endpoints to get shifts, kits, and volunteers data
app.get('/api/shifts', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM shifts ORDER BY date ASC');
    res.json(result.rows);
  } catch (err) {
    console.error('DB error fetching shifts:', err);
    res.status(500).json({ error: 'Could not fetch shifts' });
  }
});

app.get('/api/kits', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM kits ORDER BY kit_id ASC');
    res.json(result.rows);
  } catch (err) {
    console.error('DB error fetching kits:', err);
    res.status(500).json({ error: 'Could not fetch kits' });
  }
});

app.get('/api/volunteers', async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT 
        volunteer_id, first_name, last_name, dob, email, address, title
      FROM volunteers
      ORDER BY first_name
    `);
    res.json(result.rows);
  } catch (err) {
    console.error("DB error fetching volunteers:", err);
    res.status(500).json({ error: 'Could not fetch volunteers' });
  }
});

// health check and default 404
app.get('/api/health', (req, res) => {
  res.json({ status: "ok", time: new Date().toISOString() });
});

app.use((req, res) => {
  res.status(404).json({ error: "Not Found" });
});

// start the server
app.listen(port, () => {
  console.log(`Dandelions server running on http://localhost:${port}`);
});
