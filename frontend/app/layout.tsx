import "./globals.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      {/* Preload SVGs /> */}
      <head>
        <link rel="preload" as="image" href="/student.svg" type="image/svg+xml" />
        <link rel="preload" as="image" href="/unh.svg" type="image/svg+xml" />
        <link rel="preload" as="image" href="/mascot.svg" type="image/svg+xml" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
