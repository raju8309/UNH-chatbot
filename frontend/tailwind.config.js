/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        unhblue: '#003366',
        unhwhite: '#FFFFFF',
        unhaccentgrey: '#A3A9AC',
        unhaccenttan: '#D7D1C4',
        unhaccentorange: '#F77A05',
        unhaccentblue: '#263645',
      },
    },
  },
  plugins: [],
};
