-- Create Rooms table
create table rooms (
  id bigint primary key generated always as identity,
  room_number text not null unique,
  room_type text not null, -- 'Single', 'Double', 'Suite'
  price_per_night numeric not null,
  status text not null default 'available', -- 'available', 'booked', 'maintenance'
  description text
);

-- Create Bookings table
create table bookings (
  id bigint primary key generated always as identity,
  room_id bigint references rooms(id),
  guest_name text not null,
  check_in date not null,
  check_out date not null,
  created_at timestamptz default now()
);

-- Insert some dummy data
insert into rooms (room_number, room_type, price_per_night, description) values
('101', 'Single', 1000, 'Basic single room with garden view'),
('102', 'Single', 1000, 'Basic single room near elevator'),
('201', 'Double', 1800, 'Spacious double room with balcony'),
('301', 'Suite', 3500, 'Luxury suite with panoramic view');

-- Create Transactions table for Accounting
create table transactions (
  id bigint primary key generated always as identity,
  amount numeric not null, -- Positive for Income, Negative for Expense (or use separate column)
  category text not null, -- 'Income', 'Salary', 'Utility', etc.
  description text,
  created_at timestamptz default now()
);

